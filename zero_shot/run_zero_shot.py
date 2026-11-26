"""Zero-shot IGT with Bolmo 1B (Korean, Ainu).

Strategies (all prompts end at an answer label, Bolmo is a base model):
    all_in_one    one call: 'Transliteration:' line + 'Gloss:' line
    step_by_step  one call that asks for Step 1 (transliteration) then Step 2 (gloss)
    two_stage     two calls: transliteration, then gloss from the predicted transliteration
                  (the earlier "step-by-step"; Korean only)

Usage (from the repository root, on a GPU machine such as Colab):
    python zero_shot/run_zero_shot.py --lang ainu --split dev --limit 10    # smoke test
    python zero_shot/run_zero_shot.py --lang ainu --split test              # full run
    python zero_shot/run_zero_shot.py --lang jejueo --split test --shots 3  # few-shot

--shots K puts K fixed train examples (with gold transliteration and gloss) into the
all-in-one and step-by-step prompts, before the sentence to annotate; output goes to
results/few_shot/<lang>/<model>_<strategy>_<K>shot_<split>.csv. --shots 0 is zero-shot.

Writes <output_dir of the language>/<model>_<strategy>_<split>[_n<limit>].csv plus a run
metadata JSON. The prompt/parse helpers have no torch dependency, so they can be tested
without a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "zero_shot.json"
MANIFEST_PATH = ROOT / "data" / "manifest.json"

STRATEGIES = ("all_in_one", "step_by_step", "two_stage")
DEFAULT_STRATEGIES = ("all_in_one", "step_by_step")  # the two settings from the agenda

# ----------------------------------------------------------------------------- prompts


def load_prompt(rel_path: str) -> str:
    """Read a template, normalising line endings and dropping trailing whitespace,
    so the prompt ends exactly at the answer label (e.g. 'Transliteration:')."""
    text = (ROOT / rel_path).read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").rstrip()


def render(template: str, **fields: str) -> str:
    """Fill {placeholders} by plain replacement (data may contain braces).

    Prompt files are language-neutral: {language} must be filled too, either here
    (language="Korean") or beforehand with fill_language().
    """
    out = template
    for key, value in fields.items():
        out = out.replace("{" + key + "}", value)
    left = re.findall(r"\{(language|original_script|translation|predicted_transliteration)\}", out)
    if left:
        raise ValueError(f"Unfilled placeholders: {left}")
    return out


def fill_language(template: str, language_name: str) -> str:
    """Fill only {language} (e.g. 'Korean', 'Ainu'); the data fields stay as placeholders."""
    return template.replace("{language}", language_name)


def prompt_sha256(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- few-shot


def _one_line(value) -> str:
    """Collapse whitespace so an example field never breaks the line-based answer format."""
    return " ".join(str(value or "").split())


def select_examples(train_rows: list[dict], k: int, seed: int, exclude_texts=(),
                    max_text_chars: int = 60, max_gloss_chars: int = 100, min_text_words: int = 3) -> list[dict]:
    """K fixed in-context examples from the train split, the same for every evaluated row.

    A train row qualifies when all four fields are non-empty, its text is not a placeholder
    such as '[ ]', its gloss contains at least one grammatical tag (2+ capital letters, so the
    examples show what a gloss looks like), it has at least min_text_words words and is short
    (keeps the prompt short; Bolmo is byte-level), and its text is not in exclude_texts (the
    dev and test texts; otherwise the model would be shown the answer). The K rows are drawn with random.Random(seed).
    """
    import random

    excluded = {_one_line(t) for t in exclude_texts}
    pool = []
    for r in train_rows:
        text, gloss = _one_line(r.get("text")), _one_line(r.get("glosses"))
        if not all(_one_line(r.get(f)) for f in ("text", "translation", "transcription", "glosses")):
            continue
        if re.fullmatch(r"\[\s*\]", text) or text in excluded:
            continue
        if len(text) > max_text_chars or len(gloss) > max_gloss_chars or len(text.split()) < min_text_words:
            continue
        if not re.search(r"[A-Z]{2,}", gloss):
            continue
        pool.append(r)
    if len(pool) < k:
        raise ValueError(f"only {len(pool)} train rows qualify as examples, need {k}")
    return random.Random(seed).sample(pool, k)


def examples_block(examples: list[dict], language_name: str) -> str:
    """The examples in exactly the field and answer format the target sentence gets."""
    parts = []
    for e in examples:
        parts.append(
            f"Original {language_name} script: {_one_line(e['text'])}\n"
            f"English translation: {_one_line(e['translation'])}\n"
            f"Transliteration: {_one_line(e['transcription'])}\n"
            f"Gloss: {_one_line(e['glosses'])}"
        )
    return "\n\n".join(parts)


def add_examples(template: str, block: str, language_name: str) -> str:
    """Turn a language-filled zero-shot template into a few-shot prompt:

        <header and instructions of the template>
        <example 1> ... <example K>          (script, translation, Transliteration, Gloss)
        Original <language> script: {original_script}
        English translation: {translation}
        Transliteration:

    The sentence to annotate comes last, in the same format as the examples, so the prompt
    ends at the answer label right after it. (v1 put the examples before the sentence and
    left the instructions between the sentence and the label; on 2 Jejueo dev rows Bolmo
    then copied example 1 verbatim.)
    """
    fields = (f"Original {language_name} script: {{original_script}}\n"
              "English translation: {translation}\n")
    label = "\n\nTransliteration:"
    if template.count(fields) != 1 or not template.endswith(label):
        raise ValueError("template must hold the script and translation lines once and end with "
                         "'Transliteration:'")
    instructions = template.replace(fields, "")[: -len(label)].rstrip()
    return f"{instructions}\n\n{block}\n\n{fields}Transliteration:"


# ----------------------------------------------------------------------------- parsing

_GLOSS_LINE = re.compile(r"^\s*gloss\s*:\s*(.*)$", re.IGNORECASE)
_STEP2_GLOSS_LINE = re.compile(r"^\s*(?:step\s*2\s*[-:.]?\s*)?gloss\s*:\s*(.*)$", re.IGNORECASE)  # v5: "Gloss:"; v4: "Step 2 - Gloss:"
_ANY_GLOSS_LABEL = re.compile(r"^\s*(step\s*2\s*[-:.]?\s*)?gloss\s*:", re.IGNORECASE)


def _lines(completion: str) -> list[str]:
    return completion.replace("\r\n", "\n").lstrip().split("\n")


def is_complete(kind: str, completion: str) -> bool:
    """True once the completion contains every line the prompt asks for.

    kind = "single": one answer line (two-stage calls).
    kind = "all_in_one": transliteration line + gloss line (all-in-one and step-by-step).
        Done once a gloss label line is finished, or two non-empty lines are finished
        (the model may leave the transliteration empty and go straight to the gloss label).
    A line only counts as finished when a newline follows it.
    """
    if kind == "single":
        return len(_lines(completion)[:-1]) >= 1
    if kind == "all_in_one":
        finished = completion.replace("\r\n", "\n").split("\n")[:-1]
        if any(_ANY_GLOSS_LABEL.match(l) for l in finished[1:]):
            return True
        return sum(1 for l in finished if l.strip()) >= 2
    raise ValueError(kind)


def parse_single(completion: str) -> str:
    """First line of the completion, stripped (the answer after 'Label:')."""
    return _lines(completion)[0].strip()


def _parse_two_lines(completion: str, gloss_line: re.Pattern) -> tuple[str, str, str]:
    """Line 1 (right after the prompt's transliteration label) is the transliteration.
    If it is empty, the answer may start on the next non-empty line, unless that line is
    already the gloss label (the model skipped the transliteration). The next non-empty
    line must be the gloss label. Nothing is repaired.

    parse_status: ok | missing_gloss (no gloss label) | empty_answer (both empty)
                  | empty_gloss | empty_transliteration
    """
    lines = completion.replace("\r\n", "\n").split("\n")
    translit, start = lines[0].strip(), 1
    if not translit:
        for j in range(1, len(lines)):
            if lines[j].strip():
                if _ANY_GLOSS_LABEL.match(lines[j]):
                    start = j
                else:
                    translit, start = lines[j].strip(), j + 1
                break
    gloss, has_label = "", False
    for line in lines[start:]:
        if not line.strip():
            continue
        m = gloss_line.match(line)
        if m:
            gloss, has_label = m.group(1).strip(), True
        break
    if not has_label:
        status = "missing_gloss"
    elif not translit and not gloss:
        status = "empty_answer"
    elif not gloss:
        status = "empty_gloss"
    elif not translit:
        status = "empty_transliteration"
    else:
        status = "ok"
    return translit, gloss, status


def parse_all_in_one(completion: str) -> tuple[str, str, str]:
    """Return (transliteration, gloss, parse_status).

    The prompt ends with 'Transliteration:', so line 1 is the transliteration.
    The next non-empty line must start with 'Gloss:'. Nothing is repaired.
    """
    return _parse_two_lines(completion, _GLOSS_LINE)


def parse_step_by_step(completion: str) -> tuple[str, str, str]:
    """Like parse_all_in_one, for the single-call step-by-step prompt.

    v5 uses the same labels as all-in-one (prompt ends with 'Transliteration:', then a
    'Gloss:' line); the v4 label 'Step 2 - Gloss:' is accepted too.
    """
    return _parse_two_lines(completion, _STEP2_GLOSS_LINE)


# ----------------------------------------------------------------------------- generation


REQUIRED_TRANSFORMERS = "4.57.3"  # version Bolmo was tested with (model card)


def generated_part(seq: list[int], prompt_ids: list[int]) -> list[int]:
    """The newly generated ids inside the sequence a stopping criterion receives.

    Standard transformers passes prompt + generated. Bolmo's custom generate passes only
    [last prompt byte] + generated bytes (modeling_bolmo.py, generate loop), so the first
    id is dropped. Before this fix the stopper never fired on Bolmo.
    """
    if len(seq) >= len(prompt_ids) and seq[: len(prompt_ids)] == prompt_ids:
        return seq[len(prompt_ids):]
    return seq[1:]


def check_dependencies() -> None:
    """Fail early, with a fix, instead of deep inside Bolmo's remote code."""
    import importlib.util

    import transformers

    problems = []
    if importlib.util.find_spec("xlstm") is None:
        problems.append("xlstm is not installed")
    if transformers.__version__ != REQUIRED_TRANSFORMERS:
        problems.append(f"transformers {transformers.__version__} is loaded, need {REQUIRED_TRANSFORMERS}")
    if problems:
        raise RuntimeError(
            "; ".join(problems)
            + f'.\nFix: run  !pip install "transformers=={REQUIRED_TRANSFORMERS}" "xlstm==2.0.4"'
            + "  then Runtime -> Restart session, and run the notebook again from the top."
        )


class Generator:
    def __init__(self, model_id: str, revision: str, max_new_tokens: int, device: str | None = None):
        check_dependencies()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.use_stopping = True
        self._StoppingCriteria = StoppingCriteria

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _stopper(self, kind: str, prompt_ids: list[int]):
        gen = self

        class _Stop(self._StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                new = generated_part(input_ids[0].tolist(), prompt_ids)
                done = is_complete(kind, gen._decode(new))
                return gen.torch.tensor([done], dtype=gen.torch.bool, device=input_ids.device)

        return _Stop()

    def complete(self, prompt: str, kind: str) -> tuple[str, int, bool]:
        """Greedy completion. Returns (completion_text, n_new_tokens, truncated)."""
        from transformers import StoppingCriteriaList

        input_ids = self.tokenizer([prompt], return_tensors="pt")["input_ids"].to(self.device)
        kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1)
        with self.torch.no_grad():
            if self.use_stopping:
                try:
                    out = self.model.generate(
                        input_ids,
                        stopping_criteria=StoppingCriteriaList([self._stopper(kind, input_ids[0].tolist())]),
                        **kwargs,
                    )
                except (TypeError, ValueError, NotImplementedError) as err:
                    print(f"[warn] stopping_criteria not supported ({err!r}); generating full budget.")
                    self.use_stopping = False
            if not self.use_stopping:
                out = self.model.generate(input_ids, **kwargs)
        out_ids = out[0].tolist()
        in_ids = input_ids[0].tolist()
        gen_ids = out_ids[len(in_ids):] if out_ids[: len(in_ids)] == in_ids else out_ids
        text = self._decode(gen_ids)
        truncated = (not is_complete(kind, text)) and len(gen_ids) >= self.max_new_tokens
        return text, len(gen_ids), truncated


# ----------------------------------------------------------------------------- runs


def run_all_in_one(gen: Generator, rows: list[dict], template: str, on_record=None) -> list[dict]:
    out = []
    for i, r in enumerate(rows, 1):
        prompt = render(template, original_script=r["text"], translation=r["translation"])
        raw, n_new, truncated = gen.complete(prompt, "all_in_one")
        translit, gloss, status = parse_all_in_one(raw)
        out.append({
            **_gold(r),
            "pred_transliteration": translit,
            "pred_gloss": gloss,
            "parse_status": status,
            "truncated": truncated,
            "raw_output": raw,
            "n_new_tokens": n_new,
        })
        if on_record:
            on_record(out[-1])
        _progress(i, len(rows), r["id"], status)
    return out


def run_step_by_step(gen: Generator, rows: list[dict], template: str, on_record=None) -> list[dict]:
    """One call; the prompt asks for Step 1 (transliteration), then Step 2 (gloss)."""
    out = []
    for i, r in enumerate(rows, 1):
        prompt = render(template, original_script=r["text"], translation=r["translation"])
        raw, n_new, truncated = gen.complete(prompt, "all_in_one")  # same two-line stop rule
        translit, gloss, status = parse_step_by_step(raw)
        out.append({
            **_gold(r),
            "pred_transliteration": translit,
            "pred_gloss": gloss,
            "parse_status": status,
            "truncated": truncated,
            "raw_output": raw,
            "n_new_tokens": n_new,
        })
        if on_record:
            on_record(out[-1])
        _progress(i, len(rows), r["id"], status)
    return out


def run_two_stage(gen: Generator, rows: list[dict], t_step1: str, t_step2: str, on_record=None) -> list[dict]:
    """Two calls: transliteration, then gloss from the model's own predicted transliteration."""
    out = []
    for i, r in enumerate(rows, 1):
        p1 = render(t_step1, original_script=r["text"], translation=r["translation"])
        raw1, n1, trunc1 = gen.complete(p1, "single")
        translit = parse_single(raw1)
        # Step 2 always receives the model's own step-1 prediction, never gold.
        p2 = render(t_step2, original_script=r["text"], translation=r["translation"],
                    predicted_transliteration=translit)
        raw2, n2, trunc2 = gen.complete(p2, "single")
        gloss = parse_single(raw2)
        if not translit:
            status = "empty_transliteration"
        elif not gloss:
            status = "empty_gloss"
        else:
            status = "ok"
        out.append({
            **_gold(r),
            "pred_transliteration": translit,
            "pred_gloss": gloss,
            "parse_status": status,
            "truncated": trunc1 or trunc2,
            "truncated_step1": trunc1,
            "truncated_step2": trunc2,
            "raw_output_step1": raw1,
            "raw_output_step2": raw2,
            "n_new_tokens_step1": n1,
            "n_new_tokens_step2": n2,
        })
        if on_record:
            on_record(out[-1])
        _progress(i, len(rows), r["id"], status)
    return out


def _gold(r: dict) -> dict:
    return {
        "id": r["id"],
        "text": r["text"],
        "translation": r["translation"],
        "gold_transliteration": r["transcription"],
        "gold_gloss": r["glosses"],
    }


def _progress(i: int, n: int, rid: str, status: str) -> None:
    if i == 1 or i % 10 == 0 or i == n:
        print(f"  [{i}/{n}] {rid}: {status}", flush=True)


# ----------------------------------------------------------------------------- checkpoints
# With --resume, every finished row is appended to <out_csv>.partial.jsonl right away
# (flushed and fsynced), so a crash, a closed browser or a recycled Colab runtime loses at
# most the row in progress. Line 1 holds the run signature; a rerun with the same settings
# skips the rows already in the file. Greedy decoding makes the resumed rows identical to
# an uninterrupted run. Never run two processes on the same checkpoint at the same time.


def checkpoint_path(out_csv: Path) -> Path:
    return out_csv.with_suffix(".partial.jsonl")


def load_checkpoint(path: Path, signature: dict) -> dict:
    """Finished records by row id. Raises if the file belongs to a run with other settings.
    A last line cut off by a crash is ignored (that row is simply redone)."""
    done = {}
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    if not lines or not lines[0].strip():
        return done
    header = json.loads(lines[0])
    if header.get("signature") != signature:
        raise SystemExit(f"{path} was written by a run with different settings; "
                         "use another --out-dir or delete the file to start over.\n"
                         f"  file: {header.get('signature')}\n  now:  {signature}")
    for line in lines[1:]:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # partial last line from an interrupted write
        done[rec["id"]] = rec
    return done


def open_checkpoint(path: Path, signature: dict):
    """Append handle; writes the signature line when the file is new or empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    fh = open(path, "a", encoding="utf-8")
    if new:
        fh.write(json.dumps({"signature": signature}, ensure_ascii=False) + "\n")
        _sync(fh)
    else:
        with open(path, "rb") as check:  # make sure we start on a fresh line
            check.seek(-1, os.SEEK_END)
            if check.read(1) != b"\n":
                fh.write("\n")
    return fh


def append_checkpoint(fh, record: dict) -> None:
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    _sync(fh)


def _sync(fh) -> None:
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except OSError:
        pass


# ----------------------------------------------------------------------------- data + metadata


def load_rows(split: str, limit: int | None, manifest_path: Path = MANIFEST_PATH) -> list[dict]:
    """split is train / dev / test; the manifest maps it to the dataset's own split name
    (Korean: dev, Ainu: eval)."""
    from datasets import load_dataset

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    key = {"train": "train", "dev": "validation", "test": "test"}[split]
    hf_split = manifest.get("split_mapping", {}).get(key, split)
    ds = load_dataset(manifest["dataset_id"], revision=manifest["revision"], split=hf_split)
    expected = manifest["split_counts"].get(key)
    if expected is not None and len(ds) != expected:
        raise RuntimeError(f"{split} ({hf_split}) has {len(ds)} rows, manifest says {expected}")
    rows = [dict(r) for r in ds]
    return rows[:limit] if limit else rows


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _versions() -> dict:
    vers = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("torch", "transformers", "datasets", "xlstm", "pandas"):
        try:
            vers[pkg] = __import__(pkg).__version__
        except Exception:
            vers[pkg] = None
    try:
        import torch
        vers["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        vers["gpu"] = None
    return vers


def write_outputs(records: list[dict], out_csv: Path, meta: dict) -> None:
    import pandas as pd

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8")
    n = len(records)
    meta = {
        **meta,
        "n_examples": n,
        "parse_status_counts": pd.Series([r["parse_status"] for r in records]).value_counts().to_dict(),
        "truncation_rate": (sum(bool(r["truncated"]) for r in records) / n) if n else None,
    }
    out_csv.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_csv} ({n} rows)")


def main(argv: list[str] | None = None) -> list[Path]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="korean", help="a key of 'languages' in configs/zero_shot.json")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    ap.add_argument("--strategy", choices=[*STRATEGIES, "both"], default="both",
                    help="'both' = all_in_one + step_by_step (the two agenda settings)")
    ap.add_argument("--out-dir", default=None, help="default: output_dir of the language in the config")
    ap.add_argument("--shots", type=int, default=0,
                    help="few-shot: number of train examples in the prompt (0 = zero-shot, the default)")
    ap.add_argument("--resume", action="store_true",
                    help="save each finished row to <output>.partial.jsonl and skip rows already "
                         "there; rerun the same command after an interruption to continue")
    args = ap.parse_args(argv)
    if args.shots < 0:
        raise SystemExit("--shots must be 0 or more")

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.lang not in cfg["languages"]:
        raise SystemExit(f"Unknown --lang {args.lang!r}; choose from {sorted(cfg['languages'])}")
    lang_cfg = cfg["languages"][args.lang]
    language_name = lang_cfg["language_name"]
    model_cfg = cfg["models"][0]
    gen_cfg = cfg["generation"]
    manifest_path = ROOT / lang_cfg["dataset_manifest"]

    strategies = DEFAULT_STRATEGIES if args.strategy == "both" else (args.strategy,)
    if args.shots and "two_stage" in strategies:
        raise SystemExit("--shots is only implemented for all_in_one and step_by_step")
    needed = {"all_in_one": ["all_in_one"], "step_by_step": ["step_by_step"],
              "two_stage": ["step_1_transliteration", "step_2_gloss"]}

    import torch
    torch.manual_seed(cfg["seed"])

    generic = {k: load_prompt(v) for k, v in cfg["prompts"].items()}
    templates = {k: fill_language(t, language_name) for k, t in generic.items()}
    rows = load_rows(args.split, args.limit, manifest_path)
    print(f"Loaded {len(rows)} {args.lang} {args.split} rows")

    few_cfg = cfg["few_shot"]
    examples = []
    if args.shots:
        if args.split == "train":
            raise SystemExit("few-shot examples come from train, so evaluate on dev or test")
        # never show a dev or test sentence as an example; excluding both keeps the
        # examples identical for the dev smoke test and the test run
        eval_texts = [r["text"] for sp in ("dev", "test") for r in load_rows(sp, None, manifest_path)]
        examples = select_examples(load_rows("train", None, manifest_path), args.shots, few_cfg["seed"],
                                   eval_texts, few_cfg["max_text_chars"], few_cfg["max_gloss_chars"],
                                   few_cfg["min_text_words"])
        block = examples_block(examples, language_name)
        for k in ("all_in_one", "step_by_step"):
            templates[k] = add_examples(templates[k], block, language_name)
        print(f"{args.shots}-shot: train examples {[e['id'] for e in examples]}")

    gen = None  # loaded on first use, so a fully finished resume needs no GPU

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.shots:
        out_dir = ROOT / few_cfg["output_dir"].format(lang=args.lang)
    else:
        out_dir = ROOT / lang_cfg["output_dir"]
    tag = f"bolmo-{model_cfg['size'].lower()}"
    suffix = f"_n{args.limit}" if args.limit else ""
    shot_tag = f"_{args.shots}shot" if args.shots else ""
    written = []
    for strategy in strategies:
        print(f"== {strategy} ==")
        t0 = time.time()
        used = {k: templates[k] for k in needed[strategy]}
        out_csv = out_dir / f"{tag}_{strategy}{shot_tag}_{args.split}{suffix}.csv"
        signature = {
            "language": args.lang, "strategy": strategy, "split": args.split, "limit": args.limit,
            "shots": args.shots, "few_shot_examples": [e["id"] for e in examples],
            "model_revision": model_cfg["revision"], "generation": gen_cfg,
            "prompt_sha256_filled": {k: prompt_sha256(v) for k, v in used.items()},
        }
        done, fh = {}, None
        if args.resume:
            ids = [r["id"] for r in rows]
            if len(set(ids)) != len(ids):
                raise SystemExit("--resume needs unique row ids")
            ckpt = checkpoint_path(out_csv)
            if ckpt.exists():
                done = load_checkpoint(ckpt, signature)
            print(f"  resume: {len(done)}/{len(rows)} rows already in {ckpt}", flush=True)
            fh = open_checkpoint(ckpt, signature)
        todo = [r for r in rows if r["id"] not in done]
        new = []
        if todo:
            if gen is None:
                gen = Generator(model_cfg["model_id"], model_cfg["revision"], gen_cfg["max_new_tokens"])
            on_record = (lambda rec: append_checkpoint(fh, rec)) if fh else None
            if strategy == "all_in_one":
                new = run_all_in_one(gen, todo, templates["all_in_one"], on_record)
            elif strategy == "step_by_step":
                new = run_step_by_step(gen, todo, templates["step_by_step"], on_record)
            else:
                new = run_two_stage(gen, todo, templates["step_1_transliteration"],
                                    templates["step_2_gloss"], on_record)
        if fh:
            fh.close()
        by_id = {**done, **{r["id"]: r for r in new}}
        recs = [by_id[r["id"]] for r in rows]
        meta = {
            "language": args.lang,
            "strategy": strategy,
            "split": args.split,
            "limit": args.limit,
            "model_id": model_cfg["model_id"],
            "model_revision": model_cfg["revision"],
            "dataset_manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
            "prompt_version": cfg["prompt_version"],
            "language_name": language_name,
            "prompt_files": {k: cfg["prompts"][k] for k in used},
            "prompt_sha256": {k: prompt_sha256(generic[k]) for k in used},
            "prompt_sha256_filled": {k: prompt_sha256(v) for k, v in used.items()},
            "shots": args.shots,
            "few_shot_seed": few_cfg["seed"] if args.shots else None,
            "few_shot_examples": [{"id": e["id"], "text": e["text"], "translation": e["translation"],
                                   "transcription": e["transcription"], "glosses": e["glosses"]}
                                  for e in examples],
            "generation": gen_cfg,
            "stopping_criteria_used": gen.use_stopping if gen else None,
            "seed": cfg["seed"],
            "git_sha": _git_sha(),
            "runtime_seconds": round(time.time() - t0, 1),
            "resumed": bool(args.resume and done),
            "rows_from_checkpoint": len(done),
            "versions": _versions(),
        }
        write_outputs(recs, out_csv, meta)
        written.append(out_csv)
    return written


if __name__ == "__main__":
    main()
