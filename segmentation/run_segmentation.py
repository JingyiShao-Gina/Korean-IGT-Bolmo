"""Few-shot with morpheme segmentation (tutor, 1 Dec agenda): one call that asks for
Step 1 transliteration, Step 2 segmentation, Step 3 gloss.

    Transliteration: oraun cise soy ta ekan ruwe ne akusu ,
    Segmentation: oraun cise soy ta ek =an ruwe ne akusu ,
    Gloss: then house outside at come.SG=4.S INFR.EV COP then

The gold transcription is the segmented transliteration (clitics are split off with
"=", e.g. "a= kor", "ek =an"; affixes are not marked). The examples' unsegmented
Step 1 line is the gold transcription with the "=" boundaries removed ("ek =an" ->
"ekan"). The predicted Segmentation line is scored as the transliteration, against the
gold transcription, so transliteration scores stay comparable with the earlier runs;
the Step 1 line is kept in pred_transliteration_unsegmented.

Everything else is the 3-shot setup of zero_shot/run_zero_shot.py, which is reused and
not changed: the same 3 train examples (seed 42), the same generation settings
(greedy, max_new_tokens 768), --resume checkpoints, the same output columns.
--fast adds the speed-ups of fast_run/run_fast.py (loop stop, byte cap, bf16 on A100/L4).

Usage (from the repository root, on Colab):
    python segmentation/run_segmentation.py --lang ainu --split dev --limit 10   # smoke test
    python segmentation/run_segmentation.py --lang ainu --split test --resume    # full run
    python segmentation/score_segmentation.py --lang ainu                        # scores
Output: results/segmentation/<lang>/bolmo-1b_three_step_seg_<K>shot_<split>[_n<limit>].csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "zero_shot"))
sys.path.insert(0, str(ROOT / "fast_run"))
import run_zero_shot as rz  # noqa: E402

PROMPT_FILE = "segmentation/prompts/three_step.txt"
STRATEGY = "three_step_seg"
KIND = "three_step"

_SEG_LINE = re.compile(r"^\s*segmentation\s*:\s*(.*)$", re.IGNORECASE)
_GLOSS_LINE = re.compile(r"^\s*gloss\s*:\s*(.*)$", re.IGNORECASE)


# ----------------------------------------------------------------------------- examples


def unsegment(transcription: str) -> str:
    """Gold transcription without its clitic boundaries: 'a= kor' -> 'akor', 'ek =an' -> 'ekan'."""
    return re.sub(r"\s*=\s*", "", rz._one_line(transcription))


def examples_block(examples: list[dict], language_name: str) -> str:
    parts = []
    for e in examples:
        parts.append(
            f"Original {language_name} script: {rz._one_line(e['text'])}\n"
            f"English translation: {rz._one_line(e['translation'])}\n"
            f"Transliteration: {unsegment(e['transcription'])}\n"
            f"Segmentation: {rz._one_line(e['transcription'])}\n"
            f"Gloss: {rz._one_line(e['glosses'])}"
        )
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- stop + parse


def three_step_complete(completion: str) -> bool:
    """Done once a Gloss line is finished, or three non-empty lines are finished."""
    finished = completion.replace("\r\n", "\n").split("\n")[:-1]
    if any(_GLOSS_LINE.match(l) for l in finished[1:]):
        return True
    return sum(1 for l in finished if l.strip()) >= 3


_is_complete = rz.is_complete


def is_complete(kind: str, completion: str) -> bool:
    if kind == KIND:
        return three_step_complete(completion)
    return _is_complete(kind, completion)


rz.is_complete = is_complete  # the generators (original and fast) look it up here


def parse_three_step(completion: str) -> tuple[str, str, str, str]:
    """Return (transliteration, segmentation, gloss, parse_status).

    Line 1 (right after the prompt's 'Transliteration:') is the transliteration. The next
    non-empty line must start with 'Segmentation:', the one after with 'Gloss:'.
    Nothing is repaired.
    parse_status: ok | missing_segmentation | missing_gloss | empty_segmentation | empty_gloss
    """
    lines = [l for l in completion.replace("\r\n", "\n").split("\n")]
    translit = lines[0].strip()
    rest = [l for l in lines[1:] if l.strip()]
    seg, gloss, status = "", "", "ok"
    if not rest or not _SEG_LINE.match(rest[0]):
        status = "missing_segmentation"
        if rest and _GLOSS_LINE.match(rest[0]):  # skipped the segmentation step
            gloss = _GLOSS_LINE.match(rest[0]).group(1).strip()
        return translit, seg, gloss, status
    seg = _SEG_LINE.match(rest[0]).group(1).strip()
    if len(rest) < 2 or not _GLOSS_LINE.match(rest[1]):
        return translit, seg, gloss, "missing_gloss"
    gloss = _GLOSS_LINE.match(rest[1]).group(1).strip()
    if not seg:
        status = "empty_segmentation"
    elif not gloss:
        status = "empty_gloss"
    return translit, seg, gloss, status


# ----------------------------------------------------------------------------- run


def run_three_step(gen, rows: list[dict], template: str, on_record=None) -> list[dict]:
    out = []
    for i, r in enumerate(rows, 1):
        prompt = rz.render(template, original_script=r["text"], translation=r["translation"])
        raw, n_new, truncated = gen.complete(prompt, KIND)
        translit, seg, gloss, status = parse_three_step(raw)
        out.append({
            **rz._gold(r),
            "pred_transliteration": seg,  # scored against the (segmented) gold transcription
            "pred_gloss": gloss,
            "pred_transliteration_unsegmented": translit,
            "parse_status": status,
            "truncated": truncated,
            "raw_output": raw,
            "n_new_tokens": n_new,
        })
        if on_record:
            on_record(out[-1])
        rz._progress(i, len(rows), r["id"], status)
    return out


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="ainu")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shots", type=int, default=3, help="number of train examples (default 3)")
    ap.add_argument("--out-dir", default=None, help="default: results/segmentation/<lang>")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fast", action="store_true", help="use the speed-ups of fast_run/run_fast.py")
    ap.add_argument("--max-bytes", type=int, default=1000, help="with --fast: byte cap")
    ap.add_argument("--dtype", choices=["auto", "float32", "bfloat16"], default="auto", help="with --fast")
    args = ap.parse_args(argv)
    if args.shots < 1:
        raise SystemExit("segmentation needs examples: --shots must be 1 or more")

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

    generic = rz.load_prompt(PROMPT_FILE)
    rows = rz.load_rows(args.split, args.limit, manifest_path)
    eval_texts = [r["text"] for sp in ("dev", "test") for r in rz.load_rows(sp, None, manifest_path)]
    examples = rz.select_examples(rz.load_rows("train", None, manifest_path), args.shots, few_cfg["seed"],
                                  eval_texts, few_cfg["max_text_chars"], few_cfg["max_gloss_chars"],
                                  few_cfg["min_text_words"])
    template = rz.add_examples(rz.fill_language(generic, language_name),
                               examples_block(examples, language_name), language_name)
    print(f"Loaded {len(rows)} {args.lang} {args.split} rows; {args.shots}-shot examples "
          f"{[e['id'] for e in examples]}")

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "segmentation" / args.lang
    suffix = f"_n{args.limit}" if args.limit else ""
    out_csv = out_dir / f"bolmo-{model_cfg['size'].lower()}_{STRATEGY}_{args.shots}shot_{args.split}{suffix}.csv"
    signature = {
        "language": args.lang, "strategy": STRATEGY, "split": args.split, "limit": args.limit,
        "shots": args.shots, "few_shot_examples": [e["id"] for e in examples],
        "model_revision": model_cfg["revision"], "generation": gen_cfg,
        "prompt_sha256_filled": {STRATEGY: rz.prompt_sha256(template)},
    }
    t0 = time.time()
    done, fh = {}, None
    if args.resume:
        ckpt = rz.checkpoint_path(out_csv)
        if ckpt.exists():
            done = rz.load_checkpoint(ckpt, signature)
        print(f"  resume: {len(done)}/{len(rows)} rows already in {ckpt}", flush=True)
        fh = rz.open_checkpoint(ckpt, signature)
    todo = [r for r in rows if r["id"] not in done]
    new, gen = [], None
    if todo:
        gen = Gen(model_cfg["model_id"], model_cfg["revision"], gen_cfg["max_new_tokens"])
        on_record = (lambda rec: rz.append_checkpoint(fh, rec)) if fh else None
        new = run_three_step(gen, todo, template, on_record)
    if fh:
        fh.close()
    by_id = {**done, **{r["id"]: r for r in new}}
    recs = [by_id[r["id"]] for r in rows]
    if args.fast:
        for r in recs:
            r["stop_reason"] = fast.stop_reason(KIND, r["raw_output"], int(r["n_new_tokens"]),
                                                gen_cfg["max_new_tokens"])
    meta = {
        "language": args.lang, "strategy": STRATEGY, "split": args.split, "limit": args.limit,
        "model_id": model_cfg["model_id"], "model_revision": model_cfg["revision"],
        "dataset_manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "language_name": language_name,
        "prompt_files": {STRATEGY: PROMPT_FILE},
        "prompt_sha256": {STRATEGY: rz.prompt_sha256(generic)},
        "prompt_sha256_filled": {STRATEGY: rz.prompt_sha256(template)},
        "shots": args.shots, "few_shot_seed": few_cfg["seed"],
        "few_shot_examples": [{"id": e["id"], "text": e["text"], "translation": e["translation"],
                               "transcription": e["transcription"], "glosses": e["glosses"],
                               "transliteration_unsegmented": unsegment(e["transcription"])}
                              for e in examples],
        "segmentation_note": "pred_transliteration = predicted Segmentation line (scored against the gold "
                             "transcription, which marks clitics with '='); the Step 1 line is in "
                             "pred_transliteration_unsegmented.",
        "generation": gen_cfg,
        "stopping_criteria_used": gen.use_stopping if gen else None,
        "seed": cfg["seed"], "git_sha": rz._git_sha(),
        "runtime_seconds": round(time.time() - t0, 1),
        "resumed": bool(args.resume and done), "rows_from_checkpoint": len(done),
        "versions": rz._versions(),
        "runner": "segmentation/run_segmentation.py",
    }
    if args.fast:
        counts: dict[str, int] = {}
        for r in recs:
            counts[r["stop_reason"]] = counts.get(r["stop_reason"], 0) + 1
        meta["stop_reason_counts"] = counts
    rz.write_outputs(recs, out_csv, meta)
    return out_csv


if __name__ == "__main__":
    main()
