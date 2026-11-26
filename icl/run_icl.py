"""In-context learning with retrieved examples (tutor, 1 Dec agenda).

For every evaluated sentence, the K train sentences with the largest word overlap with it
are put into the prompt as solved examples (gold transliteration and gloss). K defaults
to 21 (the agenda asks for more than 20). Everything else is the few-shot setup of
zero_shot/run_zero_shot.py, which is reused and not changed: the same zero-shot
templates, example format, parsing, --resume checkpoints and output columns.

Word overlap (select_similar): the original-script text of both sentences is
NFKC-normalised (half-width katakana -> full-width), lower-cased and split on whitespace;
punctuation-only tokens are dropped. Score = number of shared word types. Ties are broken
by the number of shared English translation words (stopwords removed), then by the
shorter example, then by id, so the selection is deterministic. The candidate pool is
the train split, with the same filters as the fixed few-shot examples (all fields
present, a gloss tag, text <= 60 characters, gloss <= 100, at least 3 words) and no
dev or test text. The most similar example is placed last, right before the sentence.

Usage (from the repository root, on Colab):
    python icl/run_icl.py --lang ainu --split dev --limit 10                 # smoke test
    python icl/run_icl.py --lang ainu --split test --resume --fast           # full run
Output: results/icl/<lang>/bolmo-1b_<strategy>_icl<K>_<split>[_n<limit>].csv; each row
also has icl_example_ids (most similar last) and icl_overlap (its shared-word count).
--fast adds the speed-ups of fast_run/run_fast.py (loop stop, byte cap, bf16 on A100/L4).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "zero_shot"))
sys.path.insert(0, str(ROOT / "fast_run"))
import run_zero_shot as rz  # noqa: E402

STOPWORDS = set("""a an the and or but of to in on at for with from by as is are was were be been being
i me my you your he him his she her it its we us our they them their this that these those
not no so then there here what which who when where how all do did does have has had will would
""".split())


# ----------------------------------------------------------------------------- selection


def _tokens(text: str) -> set[str]:
    text = unicodedata.normalize("NFKC", rz._one_line(text)).lower()
    out = set()
    for tok in text.split():
        tok = tok.strip(".,!?;:\"'“”‘’()[]{}「」『』、。…*")
        if tok and re.search(r"\w", tok):
            out.add(tok)
    return out


def _en_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", rz._one_line(text).lower()) if w not in STOPWORDS}


def _filtered(train_rows, exclude_texts, few_cfg) -> list[dict]:
    """Train rows allowed as examples: the filters of rz.select_examples, without the random draw."""
    excluded = {rz._one_line(t) for t in exclude_texts}
    pool = []
    for r in train_rows:
        text, gloss = rz._one_line(r.get("text")), rz._one_line(r.get("glosses"))
        if not all(rz._one_line(r.get(f)) for f in ("text", "translation", "transcription", "glosses")):
            continue
        if re.fullmatch(r"\[\s*\]", text) or text in excluded:
            continue
        if (len(text) > few_cfg["max_text_chars"] or len(gloss) > few_cfg["max_gloss_chars"]
                or len(text.split()) < few_cfg["min_text_words"]):
            continue
        if not re.search(r"[A-Z]{2,}", gloss):
            continue
        pool.append(r)
    return pool


class Retriever:
    def __init__(self, pool: list[dict]):
        self.pool = pool
        self.src = [_tokens(r["text"]) for r in pool]
        self.en = [_en_tokens(r["translation"]) for r in pool]

    def select(self, row: dict, k: int) -> tuple[list[dict], list[int]]:
        """K most similar pool rows, least similar first (the most similar ends up last)."""
        src, en = _tokens(row["text"]), _en_tokens(row["translation"])
        keyed = []
        for i, r in enumerate(self.pool):
            keyed.append((-len(src & self.src[i]), -len(en & self.en[i]), len(r["text"]), str(r["id"]), i))
        keyed.sort()
        top = keyed[:k][::-1]
        return [self.pool[t[-1]] for t in top], [-t[0] for t in top]


# ----------------------------------------------------------------------------- run


def run(gen, rows, template, strategy, retriever, k, language_name, on_record=None) -> list[dict]:
    parse = rz.parse_all_in_one if strategy == "all_in_one" else rz.parse_step_by_step
    out = []
    for i, r in enumerate(rows, 1):
        examples, overlaps = retriever.select(r, k)
        filled = rz.add_examples(template, rz.examples_block(examples, language_name), language_name)
        prompt = rz.render(filled, original_script=r["text"], translation=r["translation"])
        raw, n_new, truncated = gen.complete(prompt, "all_in_one")  # same two-line stop rule
        translit, gloss, status = parse(raw)
        out.append({
            **rz._gold(r),
            "pred_transliteration": translit,
            "pred_gloss": gloss,
            "parse_status": status,
            "truncated": truncated,
            "raw_output": raw,
            "n_new_tokens": n_new,
            "icl_example_ids": " ".join(str(e["id"]) for e in examples),
            "icl_overlap": " ".join(map(str, overlaps)),
            "prompt_bytes": len(prompt.encode("utf-8")),
        })
        if on_record:
            on_record(out[-1])
        rz._progress(i, len(rows), r["id"], status)
    return out


def main(argv: list[str] | None = None) -> list[Path]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="ainu")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=21, help="examples per sentence (default 21)")
    ap.add_argument("--strategy", choices=["all_in_one", "step_by_step", "both"], default="both")
    ap.add_argument("--out-dir", default=None, help="default: results/icl/<lang>")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fast", action="store_true", help="use the speed-ups of fast_run/run_fast.py")
    ap.add_argument("--max-bytes", type=int, default=1000, help="with --fast: byte cap")
    ap.add_argument("--dtype", choices=["auto", "float32", "bfloat16"], default="auto", help="with --fast")
    args = ap.parse_args(argv)

    cfg = json.loads(rz.CONFIG_PATH.read_text(encoding="utf-8"))
    lang_cfg = cfg["languages"][args.lang]
    language_name = lang_cfg["language_name"]
    model_cfg, few_cfg = cfg["models"][0], cfg["few_shot"]
    gen_cfg = dict(cfg["generation"])
    manifest_path = ROOT / lang_cfg["dataset_manifest"]

    import torch
    torch.manual_seed(cfg["seed"])

    if args.fast:
        import run_fast as fast
        fast.SETTINGS.update(max_bytes=args.max_bytes, dtype=fast.resolve_dtype(args.dtype))
        gen_cfg["speedup"] = dict(fast.SETTINGS)
        Gen = fast.FastGenerator
    else:
        Gen = rz.Generator

    rows = rz.load_rows(args.split, args.limit, manifest_path)
    eval_texts = [r["text"] for sp in ("dev", "test") for r in rz.load_rows(sp, None, manifest_path)]
    pool = _filtered(rz.load_rows("train", None, manifest_path), eval_texts, few_cfg)
    retriever = Retriever(pool)
    print(f"Loaded {len(rows)} {args.lang} {args.split} rows; {len(pool)} train rows in the example pool; "
          f"K = {args.k}", flush=True)

    strategies = ("all_in_one", "step_by_step") if args.strategy == "both" else (args.strategy,)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "icl" / args.lang
    suffix = f"_n{args.limit}" if args.limit else ""
    tag = f"bolmo-{model_cfg['size'].lower()}"
    selection = {"method": "word overlap on the original script (NFKC, lower-cased, whitespace tokens), "
                           "ties: English translation word overlap, shorter text, id; most similar last",
                 "k": args.k, "pool_size": len(pool),
                 "pool_filters": {k: few_cfg[k] for k in ("max_text_chars", "max_gloss_chars", "min_text_words")}}
    gen, written = None, []
    for strategy in strategies:
        print(f"== {strategy} ==", flush=True)
        generic = rz.load_prompt(cfg["prompts"][strategy])
        template = rz.fill_language(generic, language_name)
        out_csv = out_dir / f"{tag}_{strategy}_icl{args.k}_{args.split}{suffix}.csv"
        signature = {"language": args.lang, "strategy": strategy, "split": args.split, "limit": args.limit,
                     "icl": selection, "model_revision": model_cfg["revision"], "generation": gen_cfg,
                     "prompt_sha256_filled": {strategy: rz.prompt_sha256(template)}}
        t0 = time.time()
        done, fh = {}, None
        if args.resume:
            ckpt = rz.checkpoint_path(out_csv)
            if ckpt.exists():
                done = rz.load_checkpoint(ckpt, signature)
            print(f"  resume: {len(done)}/{len(rows)} rows already in {ckpt}", flush=True)
            fh = rz.open_checkpoint(ckpt, signature)
        todo = [r for r in rows if r["id"] not in done]
        new = []
        if todo:
            if gen is None:
                gen = Gen(model_cfg["model_id"], model_cfg["revision"], gen_cfg["max_new_tokens"])
            on_record = (lambda rec: rz.append_checkpoint(fh, rec)) if fh else None
            new = run(gen, todo, template, strategy, retriever, args.k, language_name, on_record)
        if fh:
            fh.close()
        by_id = {**done, **{r["id"]: r for r in new}}
        recs = [by_id[r["id"]] for r in rows]
        meta = {
            "language": args.lang, "strategy": strategy, "split": args.split, "limit": args.limit,
            "model_id": model_cfg["model_id"], "model_revision": model_cfg["revision"],
            "dataset_manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
            "prompt_version": cfg["prompt_version"], "language_name": language_name,
            "prompt_files": {strategy: cfg["prompts"][strategy]},
            "prompt_sha256": {strategy: rz.prompt_sha256(generic)},
            "prompt_sha256_filled": {strategy: rz.prompt_sha256(template)},
            "icl": selection,
            "prompt_bytes_mean": round(sum(int(r["prompt_bytes"]) for r in recs) / len(recs), 1) if recs else None,
            "prompt_bytes_max": max(int(r["prompt_bytes"]) for r in recs) if recs else None,
            "generation": gen_cfg,
            "stopping_criteria_used": gen.use_stopping if gen else None,
            "seed": cfg["seed"], "git_sha": rz._git_sha(),
            "runtime_seconds": round(time.time() - t0, 1),
            "resumed": bool(args.resume and done), "rows_from_checkpoint": len(done),
            "versions": rz._versions(),
            "runner": "icl/run_icl.py",
        }
        if args.fast:
            counts: dict[str, int] = {}
            for r in recs:
                r["stop_reason"] = fast.stop_reason("all_in_one", r["raw_output"], int(r["n_new_tokens"]),
                                                    gen_cfg["max_new_tokens"])
                counts[r["stop_reason"]] = counts.get(r["stop_reason"], 0) + 1
            meta["stop_reason_counts"] = counts
        rz.write_outputs(recs, out_csv, meta)
        written.append(out_csv)
    return written


if __name__ == "__main__":
    main()
