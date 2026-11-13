"""Zero-shot Korean IGT with Bolmo 1B: all-in-one and step-by-step prompting.

Usage (from the repository root, on a GPU machine such as Colab):
    python zero_shot/run_zero_shot.py --split dev --limit 10      # smoke test
    python zero_shot/run_zero_shot.py --split test                # full run

Writes results/zero_shot/<model>_<strategy>_<split>[_n<limit>].csv plus a run metadata JSON.
The prompt/parse helpers have no torch dependency so they can be tested without a GPU.
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

STRATEGIES = ("all_in_one", "step_by_step")

# ----------------------------------------------------------------------------- prompts


def load_prompt(rel_path: str) -> str:
    """Read a template, normalising line endings and dropping trailing whitespace,
    so the prompt ends exactly at the answer label (e.g. 'Transliteration:')."""
    text = (ROOT / rel_path).read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").rstrip()


def render(template: str, **fields: str) -> str:
    """Fill {placeholders} by plain replacement (data may contain braces)."""
    out = template
    for key, value in fields.items():
        out = out.replace("{" + key + "}", value)
    left = re.findall(r"\{(original_script|translation|predicted_transliteration)\}", out)
    if left:
        raise ValueError(f"Unfilled placeholders: {left}")
    return out


def prompt_sha256(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- parsing

_GLOSS_LINE = re.compile(r"^\s*gloss\s*:\s*(.*)$", re.IGNORECASE)


def _lines(completion: str) -> list[str]:
    return completion.replace("\r\n", "\n").lstrip().split("\n")


def is_complete(kind: str, completion: str) -> bool:
    """True once the completion contains every line the prompt asks for.

    kind = "single": one answer line (step 1, step 2).
    kind = "all_in_one": transliteration line, then a non-empty Gloss line.
    A line only counts as finished when a newline follows it.
    """
    lines = _lines(completion)
    finished = lines[:-1]  # the last element has no newline after it yet
    if kind == "single":
        return len(finished) >= 1
    if kind == "all_in_one":
        rest = [l for l in finished[1:] if l.strip()]
        return len(finished) >= 1 and len(rest) >= 1
    raise ValueError(kind)


def parse_single(completion: str) -> str:
    """First line of the completion, stripped (the answer after 'Label:')."""
    return _lines(completion)[0].strip()


def parse_all_in_one(completion: str) -> tuple[str, str, str]:
    """Return (transliteration, gloss, parse_status).

    The prompt ends with 'Transliteration:', so line 1 is the transliteration.
    The next non-empty line must start with 'Gloss:'. Nothing is repaired.
    """
    lines = _lines(completion)
    translit = lines[0].strip()
    gloss, status = "", "missing_gloss"
    for line in lines[1:]:
        if not line.strip():
            continue
        m = _GLOSS_LINE.match(line)
        if m:
            gloss = m.group(1).strip()
            status = "ok" if gloss else "empty_gloss"
        break
    if not translit and status == "ok":
        status = "empty_transliteration"
    return translit, gloss, status


# ----------------------------------------------------------------------------- generation


class Generator:
    def __init__(self, model_id: str, revision: str, max_new_tokens: int, device: str | None = None):
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

    def _stopper(self, kind: str, prompt_len: int):
        gen = self

        class _Stop(self._StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                done = is_complete(kind, gen._decode(input_ids[0, prompt_len:].tolist()))
                return gen.torch.tensor([done], dtype=gen.torch.bool, device=input_ids.device)

        return _Stop()

    def complete(self, prompt: str, kind: str) -> tuple[str, int, bool]:
        """Greedy completion. Returns (completion_text, n_new_tokens, truncated)."""
        from transformers import StoppingCriteriaList

        input_ids = self.tokenizer([prompt], return_tensors="pt")["input_ids"].to(self.device)
        prompt_len = input_ids.shape[1]
        kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1)
        with self.torch.no_grad():
            if self.use_stopping:
                try:
                    out = self.model.generate(
                        input_ids,
                        stopping_criteria=StoppingCriteriaList([self._stopper(kind, prompt_len)]),
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


def run_step_by_step(gen: Generator, rows: list[dict], t_step1: str, t_step2: str) -> list[dict]:
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


def load_rows(split: str, limit: int | None) -> list[dict]:
    from datasets import load_dataset

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    ds = load_dataset(manifest["dataset_id"], revision=manifest["revision"], split=split)
    expected = {"train": manifest["split_counts"]["train"],
                "dev": manifest["split_counts"]["validation"],
                "test": manifest["split_counts"]["test"]}.get(split)
    if expected is not None and len(ds) != expected:
        raise RuntimeError(f"{split} has {len(ds)} rows, manifest says {expected}")
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
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    ap.add_argument("--strategy", choices=[*STRATEGIES, "both"], default="both")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "zero_shot"))
    args = ap.parse_args(argv)

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model_cfg = cfg["models"][0]
    gen_cfg = cfg["generation"]

    import torch
    torch.manual_seed(cfg["seed"])

    templates = {k: load_prompt(v) for k, v in cfg["prompts"].items()}
    rows = load_rows(args.split, args.limit)
    print(f"Loaded {len(rows)} {args.split} rows")
    gen = Generator(model_cfg["model_id"], model_cfg["revision"], gen_cfg["max_new_tokens"])

    strategies = STRATEGIES if args.strategy == "both" else (args.strategy,)
    tag = f"bolmo-{model_cfg['size'].lower()}"
    suffix = f"_n{args.limit}" if args.limit else ""
    written = []
    for strategy in strategies:
        print(f"== {strategy} ==")
        t0 = time.time()
        if strategy == "all_in_one":
            recs = run_all_in_one(gen, rows, templates["all_in_one"])
            used = {"all_in_one": templates["all_in_one"]}
        else:
            recs = run_step_by_step(gen, rows, templates["step_1_transliteration"], templates["step_2_gloss"])
            used = {k: templates[k] for k in ("step_1_transliteration", "step_2_gloss")}
        meta = {
            "strategy": strategy,
            "split": args.split,
            "limit": args.limit,
            "model_id": model_cfg["model_id"],
            "model_revision": model_cfg["revision"],
            "dataset_manifest": json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
            "prompt_version": cfg["prompt_version"],
            "prompt_sha256": {k: prompt_sha256(v) for k, v in used.items()},
            "generation": gen_cfg,
            "stopping_criteria_used": gen.use_stopping,
            "seed": cfg["seed"],
            "git_sha": _git_sha(),
            "runtime_seconds": round(time.time() - t0, 1),
            "versions": _versions(),
        }
        out_csv = Path(args.out_dir) / f"{tag}_{strategy}_{args.split}{suffix}.csv"
        write_outputs(recs, out_csv, meta)
        written.append(out_csv)
    return written


if __name__ == "__main__":
    main()
