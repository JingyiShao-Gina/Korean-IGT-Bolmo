"""Zero-shot IGT with Bolmo 1B (Korean, Ainu).

Strategies (all prompts end at an answer label, Bolmo is a base model):
    all_in_one    one call: 'Transliteration:' line + 'Gloss:' line
    step_by_step  one call that asks for Step 1 (transliteration) then Step 2 (gloss)
    two_stage     two calls: transliteration, then gloss from the predicted transliteration
                  (the earlier "step-by-step"; Korean only)

Usage (from the repository root, on a GPU machine such as Colab):
    python zero_shot/run_zero_shot.py --lang ainu --split dev --limit 10    # smoke test
    python zero_shot/run_zero_shot.py --lang ainu --split test              # full run

Writes <output_dir of the language>/<model>_<strategy>_<split>[_n<limit>].csv plus a run
metadata JSON. The prompt/parse helpers have no torch dependency, so they can be tested
without a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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


def run_all_in_one(gen: Generator, rows: list[dict], template: str) -> list[dict]:
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
        _progress(i, len(rows), r["id"], status)
    return out


def run_step_by_step(gen: Generator, rows: list[dict], template: str) -> list[dict]:
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
        _progress(i, len(rows), r["id"], status)
    return out


def run_two_stage(gen: Generator, rows: list[dict], t_step1: str, t_step2: str) -> list[dict]:
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
    args = ap.parse_args(argv)

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.lang not in cfg["languages"]:
        raise SystemExit(f"Unknown --lang {args.lang!r}; choose from {sorted(cfg['languages'])}")
    lang_cfg = cfg["languages"][args.lang]
    language_name = lang_cfg["language_name"]
    model_cfg = cfg["models"][0]
    gen_cfg = cfg["generation"]
    manifest_path = ROOT / lang_cfg["dataset_manifest"]

    strategies = DEFAULT_STRATEGIES if args.strategy == "both" else (args.strategy,)
    needed = {"all_in_one": ["all_in_one"], "step_by_step": ["step_by_step"],
              "two_stage": ["step_1_transliteration", "step_2_gloss"]}

    import torch
    torch.manual_seed(cfg["seed"])

    generic = {k: load_prompt(v) for k, v in cfg["prompts"].items()}
    templates = {k: fill_language(t, language_name) for k, t in generic.items()}
    rows = load_rows(args.split, args.limit, manifest_path)
    print(f"Loaded {len(rows)} {args.lang} {args.split} rows")
    gen = Generator(model_cfg["model_id"], model_cfg["revision"], gen_cfg["max_new_tokens"])

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / lang_cfg["output_dir"]
    tag = f"bolmo-{model_cfg['size'].lower()}"
    suffix = f"_n{args.limit}" if args.limit else ""
    written = []
    for strategy in strategies:
        print(f"== {strategy} ==")
        t0 = time.time()
        if strategy == "all_in_one":
            recs = run_all_in_one(gen, rows, templates["all_in_one"])
        elif strategy == "step_by_step":
            recs = run_step_by_step(gen, rows, templates["step_by_step"])
        else:
            recs = run_two_stage(gen, rows, templates["step_1_transliteration"], templates["step_2_gloss"])
        used = {k: templates[k] for k in needed[strategy]}
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
            "generation": gen_cfg,
            "stopping_criteria_used": gen.use_stopping,
            "seed": cfg["seed"],
            "git_sha": _git_sha(),
            "runtime_seconds": round(time.time() - t0, 1),
            "versions": _versions(),
        }
        out_csv = out_dir / f"{tag}_{strategy}_{args.split}{suffix}.csv"
        write_outputs(recs, out_csv, meta)
        written.append(out_csv)
    return written


if __name__ == "__main__":
    main()
