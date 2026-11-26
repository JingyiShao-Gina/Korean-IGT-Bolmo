"""LoRA fine-tuning of Bolmo 1B for IGT of any language in configs/zero_shot.json: all-in-one and step-by-step.

One code for everyone; --lang picks the language (ainu, tsez, jejueo, ...). Run from the repository root on a
GPU machine (Colab):

    python lora/lora_bolmo.py train   --lang ainu --strategy step_by_step --smoke   # pipeline check, not a result
    python lora/lora_bolmo.py train   --lang ainu --strategy step_by_step           # resumable
    python lora/lora_bolmo.py eval    --lang ainu --strategy step_by_step           # fill in any checkpoint without metrics
    python lora/lora_bolmo.py select  --lang ainu --strategy step_by_step           # loss-selected + chrF++-selected step
    python lora/lora_bolmo.py predict --lang ainu --strategy step_by_step --split test             # zero-shot prompt
    python lora/lora_bolmo.py predict --lang ainu --strategy step_by_step --split test --shots 3   # few-shot prompt
    python lora/lora_bolmo.py export  --lang ainu --strategy step_by_step           # small files -> results/lora/<lang>/

--strategy: all_in_one | step_by_step.   --run: a key of "runs" in configs/lora.json (baseline).

Languages, datasets, prompt files and few-shot settings are read from configs/zero_shot.json, so LoRA uses
exactly the zero-shot / few-shot prompts (zero_shot/prompts/all_in_one.txt and step_by_step.txt, {language}
filled in). Both strategies are ONE call with the same answer format; they differ only in the instructions.

Everything for one run lives in <out-root>/<lang>/<run_id>/ (on Colab: Google Drive, so a restart loses nothing):
    run_meta.json            settings, matched LoRA modules, trainable parameter count, data checks, versions
    train_log.csv            one row per optimizer step: train loss, learning rate, gradient norm
    metrics.csv              one row per checkpoint: validation loss, gloss and transliteration chrF++
    checkpoints/step-NNNNNN/ LoRA adapter + optimizer/scheduler/RNG state (resume point)
    dev_predictions/         generated dev predictions per checkpoint (source of val chrF++)
    selection.json           checkpoint chosen by min val loss and by max val gloss chrF++
    predictions/             test predictions of the selected checkpoints, same columns as zero-shot CSVs

How the model is trained (see lora/README.md for the reasoning):
- Bolmo is byte-level. Each example = <bos> + prompt bytes + target bytes + <eos>; loss only on target bytes + <eos>.
- The prompt is the zero-shot prompt (no examples); it ends with 'Transliteration:'. The target is
  ' <transliteration>\nGloss: <gloss>', which the zero-shot parsers read back.
- Bolmo's output vocabulary has two entries per byte: "byte" and "byte, then patch boundary". The target for each
  position is the next byte, shifted to its boundary variant where the model's own boundary predictor puts a
  boundary (the same patching the model uses at inference).
- `predict --shots K` tests the same adapter with the few-shot prompt: the K train examples are chosen exactly as
  in `zero_shot/run_zero_shot.py --shots K` (same function, seed and filters).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "zero_shot"))
import run_zero_shot as rz  # noqa: E402  (prompts, few-shot examples, parsing, data loading, versions)

CONFIG_PATH = ROOT / "configs" / "lora.json"
STRATEGIES = ("all_in_one", "step_by_step")
PARSERS = {"all_in_one": rz.parse_all_in_one, "step_by_step": rz.parse_step_by_step}
COLAB_DRIVE = Path("/content/drive/MyDrive")

# ============================================================================ config


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_zs_config() -> dict:
    """configs/zero_shot.json: languages, datasets, prompt files and few-shot settings (shared with zero-shot)."""
    return json.loads(rz.CONFIG_PATH.read_text(encoding="utf-8"))


def languages() -> list[str]:
    return sorted(load_zs_config()["languages"])


def lang_info(lang: str) -> dict:
    """{'lang', 'language_name', 'manifest_path'} of a key of 'languages' in configs/zero_shot.json."""
    langs = load_zs_config()["languages"]
    if lang not in langs:
        raise SystemExit(f"Unknown --lang {lang!r}; choose from {sorted(langs)}")
    return {"lang": lang, "language_name": langs[lang]["language_name"],
            "manifest_path": ROOT / langs[lang]["dataset_manifest"]}


def fmt_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")  # 0.0002 -> 2e-4


def run_settings(cfg: dict, run: str, strategy: str, lang: str, seed: int | None = None) -> dict:
    """Baseline adapter/training settings with the run's overrides applied."""
    if run not in cfg["runs"]:
        raise SystemExit(f"Unknown --run {run!r}; choose from {list(cfg['runs'])}")
    if strategy not in STRATEGIES:
        raise SystemExit(f"Unknown --strategy {strategy!r}; choose from {STRATEGIES}")
    lang_info(lang)
    adapter = dict(cfg["adapter"])
    training = dict(cfg["training"])
    for key, value in cfg["runs"][run].items():
        if key in adapter:
            adapter[key] = value
        elif key in training:
            training[key] = value
        else:
            raise SystemExit(f"Run {run!r} overrides unknown setting {key!r}")
    if seed is not None:
        training["seed"] = seed
    size = cfg["model"]["size"].lower()
    run_id = (f"bolmo-{size}_{lang}_{strategy}_r{adapter['r']}"
              f"_lr{fmt_lr(training['learning_rate'])}_s{training['seed']}")
    return {"run": run, "lang": lang, "strategy": strategy, "run_id": run_id,
            "adapter": adapter, "training": training}


def default_out_root(cfg: dict) -> Path:
    if COLAB_DRIVE.is_dir():
        return Path(cfg["output_root_colab"])
    return ROOT / "outputs" / "lora"  # git-ignored local fallback


def run_path(args, cfg: dict, s: dict) -> Path:
    root = Path(args.out_root) if args.out_root else default_out_root(cfg)
    return root / s["lang"] / (s["run_id"] + ("_smoke" if args.smoke else ""))


def load_templates(lang: str) -> dict[str, str]:
    """The zero-shot prompt of each strategy with {language} filled in (same files and function as zero-shot)."""
    zs, info = load_zs_config(), lang_info(lang)
    return {k: rz.fill_language(rz.load_prompt(zs["prompts"][k]), info["language_name"]) for k in STRATEGIES}


def few_shot_examples(lang: str, k: int) -> list[dict]:
    """The K fixed train examples that `run_zero_shot.py --lang <lang> --shots K` uses (same seed and filters)."""
    zs, info = load_zs_config(), lang_info(lang)
    few = zs["few_shot"]
    eval_texts = [r["text"] for sp in ("dev", "test") for r in rz.load_rows(sp, None, info["manifest_path"])]
    return rz.select_examples(rz.load_rows("train", None, info["manifest_path"]), k, few["seed"], eval_texts,
                              few["max_text_chars"], few["max_gloss_chars"], few["min_text_words"])


def prompt_fn(templates: dict[str, str], strategy: str, language_name: str, examples: list[dict] | None = None):
    """row -> prompt. Without examples: the zero-shot prompt. With examples: the few-shot prompt built exactly
    like run_zero_shot.py does. (Per-row example selection, e.g. by word overlap, plugs in here.)"""
    template = templates[strategy]
    if examples:
        template = rz.add_examples(template, rz.examples_block(examples, language_name), language_name)
    # a missing field (None in some datasets) becomes an empty string, as in the zero-shot CSVs
    return lambda r: rz.render(template, original_script=r["text"] or "", translation=r["translation"] or "")


# ============================================================================ examples (no torch)


def answer_fields(r: dict) -> tuple[str, str]:
    """Gold transliteration and gloss on one line each (the answer format is two lines; uncleaned
    data can contain line breaks or repeated spaces)."""
    return rz._one_line(r.get("transcription")), rz._one_line(r.get("glosses"))


def usable(r: dict) -> bool:
    """A row can be a training / validation-loss example only if its input (script, translation) and both
    answer fields are non-empty."""
    return bool(rz._one_line(r.get("text"))) and bool(rz._one_line(r.get("translation"))) and all(answer_fields(r))


def build_examples(rows: list[dict], strategy: str, make_prompt) -> list[dict]:
    """Supervised (prompt, target) pairs, one per usable row, for both strategies.

    The prompt (zero-shot prompt of the strategy) ends with 'Transliteration:'; the target continues it with
    the transliteration line and the 'Gloss:' line, which the strategy's zero-shot parser reads back.
    Rows with an empty transliteration or gloss are left out (see usable()).
    """
    if strategy not in STRATEGIES:
        raise ValueError(strategy)
    out = []
    for r in rows:
        if not usable(r):
            continue
        translit, gloss = answer_fields(r)
        out.append({"row_id": r["id"], "kind": strategy, "prompt": make_prompt(r),
                    "target": f" {translit}\nGloss: {gloss}"})
    return out


def example_bytes(ex: dict) -> int:
    """Encoded length: <bos> + prompt + target + <eos> (Bolmo: one id per byte)."""
    return 2 + len(ex["prompt"].encode("utf-8")) + len(ex["target"].encode("utf-8"))


def encode_example(prompt: str, target: str, *, bos: int, eos: int, offset: int) -> dict:
    """Byte ids for <bos> prompt target <eos>, and how many leading ids are prompt (not trained on)."""
    p = [offset + b for b in prompt.encode("utf-8")]
    t = [offset + b for b in target.encode("utf-8")]
    ids = [bos] + p + t + [eos]
    return {"ids": ids, "n_prompt": 1 + len(p), "n_target": len(t) + 1}


def encode_prompt(prompt: str, *, bos: int, offset: int) -> list[int]:
    return [bos] + [offset + b for b in prompt.encode("utf-8")]


def eval_steps(n_examples: int, epochs: float, accum: int, every_epochs: float) -> tuple[int, list[int]]:
    """Total optimizer steps and the steps at which to save + evaluate (every `every_epochs`, plus the last)."""
    total = math.ceil(n_examples * epochs / accum)
    per = n_examples * every_epochs / accum
    steps = sorted({min(total, max(1, round(k * per))) for k in range(1, math.ceil(epochs / every_epochs) + 1)})
    if steps[-1] != total:
        steps.append(total)
    return total, steps


def micro_order(n_examples: int, epochs: int, seed: int) -> list[int]:
    """Deterministic example order: a fresh seeded shuffle for each epoch, concatenated."""
    order = []
    for epoch in range(math.ceil(epochs)):
        idx = list(range(n_examples))
        random.Random(seed * 1000 + epoch).shuffle(idx)
        order.extend(idx)
    return order[: int(round(n_examples * epochs))]


def select_checkpoints(metrics: list[dict]) -> dict:
    """Selection A: min val_loss. Selection B: max val_gloss_chrfpp. Ties -> earlier step."""
    rows = sorted(metrics, key=lambda m: int(m["step"]))
    if not rows:
        raise SystemExit("metrics.csv is empty - train first")
    best_loss = min(rows, key=lambda m: (float(m["val_loss"]), int(m["step"])))
    best_chrf = min(rows, key=lambda m: (-float(m["val_gloss_chrfpp"]), int(m["step"])))
    pick = lambda m: {k: m[k] for k in ("step", "epoch", "val_loss", "val_gloss_chrfpp", "val_transliteration_chrfpp")}
    return {"loss": pick(best_loss), "chrf": pick(best_chrf), "same_checkpoint": best_loss["step"] == best_chrf["step"]}


def is_truncated(ended: bool, text: str, parse_kind: str) -> bool:
    """A generation stops for one of three reasons: <eos>, a complete answer, or the byte budget.
    Neither of the first two -> it hit the budget. Do not compare n_new with the budget: the returned
    outputs stop at budget - 1 bytes (511 / 255 in every test file), so the old `n_new >= max_new`
    check never fired and all truncation flags up to 16 Nov are False."""
    return (not ended) and not rz.is_complete(parse_kind, text)


# ============================================================================ metrics


def chrf_scores(records: list[dict], chrf_cfg: dict) -> dict:
    """Corpus chrF++ for gloss and transliteration; unusable predictions count as empty strings."""
    from sacrebleu.metrics import CHRF

    def score(pred_key, gold_key):
        metric = CHRF(**chrf_cfg)
        hyps = [str(r.get(pred_key) or "") for r in records]
        refs = [str(r[gold_key]) for r in records]
        return metric.corpus_score(hyps, [refs]).score, str(metric.get_signature())

    g, sig = score("pred_gloss", "gold_gloss")
    t, _ = score("pred_transliteration", "gold_transliteration")
    return {"gloss_chrfpp": round(g, 4), "transliteration_chrfpp": round(t, 4), "chrf_signature": sig}


# ============================================================================ model


class Bolmo:
    """Bolmo-1B + LoRA adapter, with the loss and generation this project needs."""

    def __init__(self, cfg: dict, settings: dict | None, *, adapter_dir: Path | None = None,
                 trainable: bool = False, device: str | None = None):
        rz.check_dependencies()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
            self.dtype = torch.bfloat16  # A100 / L4
        else:
            self.dtype = torch.float32   # T4 has no native bf16
        m = cfg["model"]
        self.tok = AutoTokenizer.from_pretrained(m["model_id"], revision=m["revision"], trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            m["model_id"], revision=m["revision"], trust_remote_code=True, torch_dtype=self.dtype
        ).to(self.device)
        self._init_from_base(base, settings, adapter_dir, trainable)

    @classmethod
    def from_model(cls, cfg, base, tok, settings, *, adapter_dir=None, trainable=False, device="cpu"):
        """Wrap an already-built model (used by the CPU tiny-model test)."""
        import torch

        self = cls.__new__(cls)
        self.torch, self.cfg, self.device, self.dtype, self.tok = torch, cfg, device, base.dtype, tok
        self._init_from_base(base, settings, adapter_dir, trainable)
        return self

    def _init_from_base(self, base, settings, adapter_dir, trainable):
        from peft import LoraConfig, PeftModel, get_peft_model

        self.base = base
        self.core = base.model  # BolmoModel: local_encoder, layers, local_decoder
        t = self.core.tokenizer
        self.bos, self.eos, self.pad, self.offset = t.bos_token_id, t.eos_token_id, t.pad_token_id, t.offset
        self.boundary_offset = t.offset + 256
        chunks = [mod.mlstm_backend.config.chunk_size for mod in base.modules() if hasattr(mod, "mlstm_backend")]
        self.chunk_size = math.lcm(*chunks) if chunks else 1
        for p in base.parameters():
            p.requires_grad_(False)

        if adapter_dir is not None:
            self.peft = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=trainable)
            self.lora_modules = sorted({n.split(".lora_")[0] for n, _ in base.named_modules() if ".lora_A" in n})
        elif settings is not None:
            a = settings["adapter"]
            regex = a["target_modules_regex"]
            self.lora_modules = [n for n, _ in base.named_modules() if re.fullmatch(regex, n)]
            if not self.lora_modules:
                raise RuntimeError(f"LoRA target regex {regex!r} matched no module - check the architecture")
            lcfg = LoraConfig(r=a["r"], lora_alpha=a["lora_alpha"], lora_dropout=a["lora_dropout"],
                              bias=a["bias"], target_modules=regex)
            self.peft = get_peft_model(base, lcfg)
        else:
            self.peft = None
            self.lora_modules = []
        # LoRA layers are injected into `base` in place, so calling `base` runs base + adapter.
        # (PEFT's own forward would pass labels/attention kwargs that Bolmo's forward does not take.)

    # ------------------------------------------------------------------ helpers

    def trainable_parameters(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.base.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.base.parameters())
        return trainable, total

    def reset_cache(self) -> None:
        """Bolmo's generate() leaves inference caches switched on; a later full forward would use them."""
        self.core.local_encoder.free_inference_cache()
        self.core.local_decoder.free_inference_cache()

    def check_encoding(self, prompt: str, target: str) -> None:
        """Our byte encoding must equal the tokenizer's (bos + utf-8 bytes)."""
        ours = encode_example(prompt, target, bos=self.bos, eos=self.eos, offset=self.offset)["ids"][:-1]
        theirs = self.tok(prompt + target)["input_ids"]
        if list(theirs) != ours:
            raise RuntimeError("Byte encoding differs from the Bolmo tokenizer - do not train")

    # ------------------------------------------------------------------ loss

    def example_loss(self, ex: dict):
        """Summed cross-entropy over the target bytes (+ eos) of one encoded example."""
        torch = self.torch
        n = len(ex["ids"])
        # The xLSTM kernels in train mode need a length divisible by their chunk size: right-pad.
        # Every layer is causal, so the padding cannot change the positions we score.
        width = math.ceil(n / self.chunk_size) * self.chunk_size
        pad = [self.pad] * (width - n)
        ids = torch.tensor([ex["ids"] + pad], device=self.device)
        expanded = torch.tensor([self.core.tokenizer.expand_byte_ids(ex["ids"]) + pad], device=self.device)
        self.reset_cache()
        with torch.no_grad():
            bmask = self.core.prefill_boundary_prediction_forward(ids, expanded_input_ids=expanded)
        out = self.base(input_ids=ids, expanded_input_ids=expanded, boundary_mask=bmask, use_cache=False)
        logits = out.logits[0, :n - 1].float()                    # position t predicts byte t+1
        target = ids[0, 1:n] + self.boundary_offset * bmask[0, 1:n].long()
        start = ex["n_prompt"] - 1                                  # first prediction of a target byte
        loss_sum = torch.nn.functional.cross_entropy(logits[start:], target[start:], reduction="sum")
        return loss_sum, len(ex["ids"]) - ex["n_prompt"]

    def validation_loss(self, encoded: list[dict]) -> dict:
        """Teacher-forced token-level mean loss, overall and per task kind. No gradients."""
        torch = self.torch
        self.base.eval()
        sums, counts = {}, {}
        with torch.no_grad():
            for ex in encoded:
                s, n = self.example_loss(ex)
                for k in ("all", ex["kind"]):
                    sums[k] = sums.get(k, 0.0) + s.item()
                    counts[k] = counts.get(k, 0) + n
        return {k: sums[k] / counts[k] for k in sums}

    # ------------------------------------------------------------------ generation

    def _stopper(self, kind: str, max_new: int):
        from transformers import StoppingCriteria

        gen = self
        parse_kind = "all_in_one"  # both strategies: transliteration line + gloss line

        class _Stop(StoppingCriteria):
            # Bolmo calls this per example with [last prompt byte, generated bytes...]
            def __call__(self, input_ids, scores, **kwargs):
                new = input_ids[0, 1:].tolist()
                done = len(new) >= max_new or rz.is_complete(parse_kind, gen.tok.decode(new, skip_special_tokens=True))
                return gen.torch.tensor([done], dtype=gen.torch.bool, device=input_ids.device)

        return _Stop()

    def generate(self, prompts: list[str], kind: str, max_new: int, batch_size: int) -> list[tuple[str, int, bool]]:
        """Greedy completions -> (text, n_new_bytes, truncated), in input order. Batches by prompt length."""
        from transformers import StoppingCriteriaList

        torch = self.torch
        self.base.eval()
        enc = [encode_prompt(p, bos=self.bos, offset=self.offset) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: -len(enc[i]))
        results: list = [None] * len(prompts)
        parse_kind = "all_in_one"
        i, bs = 0, max(1, batch_size)
        while i < len(order):
            chunk = order[i:i + bs]
            width = max(len(enc[j]) for j in chunk)
            batch = torch.tensor([[self.pad] * (width - len(enc[j])) + enc[j] for j in chunk], device=self.device)
            try:
                with torch.no_grad():
                    out = self.base.generate(
                        batch, stopping_criteria=StoppingCriteriaList([self._stopper(kind, max_new)]),
                        max_new_tokens=max_new + 16, do_sample=False, num_beams=1,
                    )
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                torch.cuda.empty_cache()
                print(f"  [oom] generation batch size -> {bs}", flush=True)
                continue
            finally:
                self.reset_cache()
            for row, j in zip(out.tolist(), chunk):
                k = 0
                while k < len(row) and row[k] == self.pad:
                    k += 1
                row = row[k:]
                if row[:len(enc[j])] != enc[j]:
                    raise RuntimeError("generate() output does not start with the prompt")
                new = row[len(enc[j]):]
                ended = bool(new) and new[-1] == self.eos
                text = self.tok.decode(new, skip_special_tokens=True)
                n_new = len(new) - (1 if ended else 0)
                truncated = is_truncated(ended, text, parse_kind)
                results[j] = (text, n_new, truncated)
            i += len(chunk)
        return results

    def predict_rows(self, rows: list[dict], strategy: str, make_prompt, gen_cfg: dict) -> list[dict]:
        """One greedy call per row; same columns as the zero-shot CSVs of the strategy."""
        prompts = [make_prompt(r) for r in rows]
        outs = self.generate(prompts, "all_in_one", gen_cfg["max_new_bytes"], gen_cfg["batch_size"])
        recs = []
        for r, (raw, n_new, trunc) in zip(rows, outs):
            translit, gloss, status = PARSERS[strategy](raw)
            recs.append({**rz._gold(r), "pred_transliteration": translit, "pred_gloss": gloss,
                         "parse_status": status, "truncated": trunc, "raw_output": raw, "n_new_tokens": n_new})
        return recs


# ============================================================================ run directory


class RunDir:
    METRIC_FIELDS = ["step", "epoch", "val_loss", "val_gloss_chrfpp", "val_transliteration_chrfpp", "parse_failure_rate", "truncation_rate",
                     "n_dev", "eval_seconds", "chrf_signature"]
    LOG_FIELDS = ["step", "epoch", "train_loss", "learning_rate", "grad_norm", "elapsed_seconds"]

    def __init__(self, path: Path):
        self.path = path
        self.ckpt_root = path / "checkpoints"
        self.dev_pred = path / "dev_predictions"
        self.pred = path / "predictions"
        for d in (self.path, self.ckpt_root, self.dev_pred, self.pred):
            d.mkdir(parents=True, exist_ok=True)

    def ckpt(self, step: int) -> Path:
        return self.ckpt_root / f"step-{step:06d}"

    def checkpoints(self) -> list[int]:
        return sorted(int(p.name.split("-")[1]) for p in self.ckpt_root.glob("step-*")
                      if not p.name.endswith(".tmp") and (p / "trainer_state.pt").exists())

    @property
    def resume_dir(self) -> Path:
        return self.path / "resume"

    def resume_step(self) -> int | None:
        """Step of the quick resume save, if a complete one exists."""
        f = self.resume_dir / "trainer_state.pt"
        if not f.exists():
            return None
        import torch
        return int(torch.load(f, map_location="cpu", weights_only=False)["step"])

    def _read(self, name: str) -> list[dict]:
        p = self.path / name
        if not p.exists():
            return []
        with p.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _append(self, name: str, fields: list[str], row: dict) -> None:
        p = self.path / name
        new = not p.exists()
        with p.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k) for k in fields})

    def metrics(self) -> list[dict]:
        return self._read("metrics.csv")

    def add_metrics(self, row: dict) -> None:
        self._append("metrics.csv", self.METRIC_FIELDS, row)

    def log_step(self, row: dict) -> None:
        self._append("train_log.csv", self.LOG_FIELDS, row)

    def trim_log(self, max_step: int) -> None:
        """After resuming from a checkpoint, drop log rows of steps that will be redone."""
        rows = [r for r in self._read("train_log.csv") if int(r["step"]) <= max_step]
        p = self.path / "train_log.csv"
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.LOG_FIELDS)
            w.writeheader()
            w.writerows(rows)


# ============================================================================ commands


def _rows(lang: str, split: str, limit: int | None) -> list[dict]:
    return rz.load_rows(split, limit, lang_info(lang)["manifest_path"])


def prepare_examples(rows: list[dict], strategy: str, make_prompt, max_bytes: int) -> tuple[list[dict], dict]:
    """Training / validation-loss examples that fit max_bytes, and a report of what was left out and why
    (nothing is truncated silently)."""
    examples = build_examples(rows, strategy, make_prompt)
    kept = [ex for ex in examples if example_bytes(ex) <= max_bytes]
    report = {
        "n_rows": len(rows),
        "n_examples": len(kept),
        "dropped_empty_field": [r["id"] for r in rows if not usable(r)],
        "dropped_too_long": [ex["row_id"] for ex in examples if example_bytes(ex) > max_bytes],
        "max_example_bytes": max((example_bytes(ex) for ex in examples), default=0),
        "max_seq_bytes": max_bytes,
    }
    return kept, report


def _encode_all(model: Bolmo, examples: list[dict]) -> list[dict]:
    return [{**encode_example(ex["prompt"], ex["target"], bos=model.bos, eos=model.eos, offset=model.offset),
             "kind": ex["kind"], "row_id": ex["row_id"]} for ex in examples]


def _print_report(name: str, rep: dict) -> None:
    print(f"{name}: {rep['n_examples']} examples from {rep['n_rows']} rows; left out "
          f"{len(rep['dropped_empty_field'])} with an empty script/translation/transliteration/gloss, "
          f"{len(rep['dropped_too_long'])} longer than {rep['max_seq_bytes']} bytes "
          f"(longest {rep['max_example_bytes']})")
    if rep["n_rows"] and len(rep["dropped_too_long"]) > 0.01 * rep["n_rows"]:
        print(f"  WARNING: more than 1% of {name} rows are too long - raise training.max_seq_bytes")


def evaluate_checkpoint(model: Bolmo, run: RunDir, step: int, epoch: float, dev_rows, dev_enc, strategy,
                        make_prompt, cfg) -> dict:
    """Validation loss (teacher-forced, usable dev rows) + chrF++ of generated answers (ALL dev rows)."""
    t0 = time.time()
    losses = model.validation_loss(dev_enc)
    recs = model.predict_rows(dev_rows, strategy, make_prompt, cfg["generation"])
    scores = chrf_scores(recs, cfg["chrf"])
    rz.write_outputs(recs, run.dev_pred / f"step-{step:06d}.csv",
                     {"split": "dev", "step": step, "epoch": epoch, **scores})
    n = len(recs)
    row = {
        "step": step, "epoch": round(epoch, 4),
        "val_loss": round(losses["all"], 6),
        "val_gloss_chrfpp": scores["gloss_chrfpp"],
        "val_transliteration_chrfpp": scores["transliteration_chrfpp"],
        "parse_failure_rate": round(sum(r["parse_status"] != "ok" for r in recs) / n, 4),
        "truncation_rate": round(sum(bool(r["truncated"]) for r in recs) / n, 4),
        "n_dev": n, "eval_seconds": round(time.time() - t0, 1), "chrf_signature": scores["chrf_signature"],
    }
    run.add_metrics(row)
    print(f"  [eval step {step}] val_loss={row['val_loss']:.4f}  gloss chrF++={row['val_gloss_chrfpp']:.2f}  "
          f"translit chrF++={row['val_transliteration_chrfpp']:.2f}  ({row['eval_seconds']}s)", flush=True)
    return row


def _zero_shot_prompt(lang: str, strategy: str):
    return prompt_fn(load_templates(lang), strategy, lang_info(lang)["language_name"])


def cmd_train(args, cfg) -> Path:
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import get_linear_schedule_with_warmup

    s = run_settings(cfg, args.run, args.strategy, args.lang, args.seed)
    tr = s["training"]
    run = RunDir(run_path(args, cfg, s))
    run_id = run.path.name
    print(f"Run {run_id} -> {run.path}")

    seed = tr["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    make_prompt = _zero_shot_prompt(args.lang, args.strategy)
    train_rows = _rows(args.lang, "train", 32 if args.smoke else None)
    dev_rows = _rows(args.lang, "dev", 4 if args.smoke else None)
    train_ex, train_rep = prepare_examples(train_rows, args.strategy, make_prompt, tr["max_seq_bytes"])
    dev_ex, dev_rep = prepare_examples(dev_rows, args.strategy, make_prompt, tr["max_seq_bytes"])
    _print_report("train", train_rep)
    _print_report("dev (validation loss; chrF++ uses all dev rows)", dev_rep)

    model = Bolmo(cfg, s, trainable=True)
    trainable, total = model.trainable_parameters()
    print(f"LoRA on {len(model.lora_modules)} modules; trainable {trainable:,} / {total:,} parameters; dtype {model.dtype}")
    model.check_encoding(train_ex[0]["prompt"], train_ex[0]["target"])
    train_enc = _encode_all(model, train_ex)
    dev_enc = _encode_all(model, dev_ex)

    accum = tr["gradient_accumulation_steps"]
    epochs = tr["num_train_epochs"]
    if args.smoke:
        accum, epochs = 4, 8 / len(train_enc)  # 2 optimizer steps on 8 examples
    total_steps, ckpt_steps = eval_steps(len(train_enc), epochs, accum, tr["eval_every_epochs"] if not args.smoke else epochs)
    order = micro_order(len(train_enc), epochs, seed)
    print(f"{len(train_enc)} train / {len(dev_enc)} dev examples; {total_steps} optimizer steps; "
          f"checkpoints at {ckpt_steps}")

    meta_path = run.path / "run_meta.json"
    if not meta_path.exists():
        zs, info = load_zs_config(), lang_info(args.lang)
        meta = {
            **s, "run_id": run_id, "smoke": args.smoke, "model": cfg["model"],
            "language_name": info["language_name"],
            "dataset_manifest": json.loads(info["manifest_path"].read_text(encoding="utf-8")),
            "prompt_version": zs["prompt_version"], "prompt_file": zs["prompts"][args.strategy],
            "prompt_sha256": rz.prompt_sha256(load_templates(args.lang)[args.strategy]),
            "lora_modules": model.lora_modules, "trainable_parameters": trainable, "total_parameters": total,
            "dtype": str(model.dtype), "n_train_examples": len(train_enc), "n_dev_examples": len(dev_enc),
            "n_dev_rows_generation": len(dev_rows), "train_data": train_rep, "dev_data": dev_rep,
            "effective_batch_size": accum * tr["per_device_train_batch_size"], "total_optimizer_steps": total_steps,
            "checkpoint_steps": ckpt_steps, "generation": cfg["generation"], "chrf": cfg["chrf"],
            "git_sha": rz._git_sha(), "versions": rz._versions(),
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    params = [p for p in model.base.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tr["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tr["learning_rate"])
    sched = get_linear_schedule_with_warmup(opt, math.ceil(tr["warmup_ratio"] * total_steps), total_steps)
    if tr["gradient_checkpointing"]:
        model.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def load_adapter(src: Path) -> None:
        set_peft_model_state_dict(model.peft, load_file(str(src / "adapter_model.safetensors"), device=model.device))

    def load_state(src: Path) -> dict:
        return torch.load(src / "trainer_state.pt", map_location="cpu", weights_only=False)

    def save_state(dst: Path, step: int, epoch: float, t_start: float) -> None:
        """Adapter + optimizer/scheduler/RNG state, written to a temp folder and renamed when complete."""
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        model.peft.save_pretrained(str(tmp))
        torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step, "epoch": epoch,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "python_rng": random.getstate(), "elapsed_seconds": time.time() - t_start},
                   tmp / "trainer_state.pt")
        shutil.rmtree(dst, ignore_errors=True)
        tmp.rename(dst)

    # ---- resume: from the newest of the evaluated checkpoints and the quick resume save
    step, elapsed0 = 0, 0.0
    evaluated = {int(m["step"]) for m in run.metrics()}
    for st in run.checkpoints():  # a session died during this checkpoint's evaluation
        if st not in evaluated:
            print(f"Evaluating checkpoint {st}, which has no metrics yet")
            load_adapter(run.ckpt(st))
            evaluate_checkpoint(model, run, st, load_state(run.ckpt(st))["epoch"], dev_rows, dev_enc,
                                args.strategy, make_prompt, cfg)
    sources = [(st, run.ckpt(st)) for st in run.checkpoints()]
    quick = run.resume_step()
    if quick is not None:
        sources.append((quick, run.resume_dir))
    if sources:
        step, src = max(sources, key=lambda x: x[0])
        load_adapter(src)
        state = load_state(src)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        elapsed0 = state.get("elapsed_seconds", 0.0)
        run.trim_log(step)
        print(f"Resumed from step {step} ({'quick resume save' if src == run.resume_dir else 'checkpoint'})")

    resume_every = 0 if args.smoke else int(tr.get("resume_every_steps", 0))
    t_start = time.time() - elapsed0
    while step < total_steps:
        model.base.train()
        group = order[step * accum:(step + 1) * accum]
        n_tok = sum(train_enc[i]["n_target"] for i in group)
        loss_total = 0.0
        for i in group:
            loss_sum, _ = model.example_loss(train_enc[i])
            (loss_sum / n_tok).backward()
            loss_total += loss_sum.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, tr["max_grad_norm"]).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        epoch = min(step * accum, len(order)) / len(train_enc)
        lr_used = sched.get_last_lr()[0]
        run.log_step({"step": step, "epoch": round(epoch, 4), "train_loss": round(loss_total / n_tok, 6),
                      "learning_rate": lr_used, "grad_norm": round(grad_norm, 4),
                      "elapsed_seconds": round(time.time() - t_start, 1)})
        if step % 10 == 0 or step == 1:
            print(f"  step {step}/{total_steps}  epoch {epoch:.2f}  loss {loss_total / n_tok:.4f}  "
                  f"lr {lr_used:.2e}  ({time.time() - t_start:.0f}s)", flush=True)
        if step in ckpt_steps:
            save_state(run.ckpt(step), step, epoch, t_start)  # a checkpoint only counts once it is complete
            evaluate_checkpoint(model, run, step, epoch, dev_rows, dev_enc, args.strategy, make_prompt, cfg)
        elif resume_every and step % resume_every == 0:
            # Quick resume point only (not evaluated, not a selection candidate): limits what a disconnect costs.
            save_state(run.resume_dir, step, epoch, t_start)
    print(f"Training done: {run.path}")
    return run.path


def _load_for_checkpoint(cfg, run: RunDir, step: int) -> Bolmo:
    return Bolmo(cfg, None, adapter_dir=run.ckpt(step), trainable=False)


def cmd_eval(args, cfg) -> None:
    """Evaluate saved checkpoints that have no metrics row yet (e.g. after a crash)."""
    s = run_settings(cfg, args.run, args.strategy, args.lang, args.seed)
    run = RunDir(run_path(args, cfg, s))
    make_prompt = _zero_shot_prompt(args.lang, args.strategy)
    dev_rows = _rows(args.lang, "dev", 4 if args.smoke else None)
    dev_ex, _ = prepare_examples(dev_rows, args.strategy, make_prompt, s["training"]["max_seq_bytes"])
    evaluated = {int(m["step"]) for m in run.metrics()}
    todo = [st for st in run.checkpoints() if st not in evaluated]
    print(f"Checkpoints without metrics: {todo}")
    import torch
    for step in todo:
        model = _load_for_checkpoint(cfg, run, step)
        state = torch.load(run.ckpt(step) / "trainer_state.pt", map_location="cpu", weights_only=False)
        evaluate_checkpoint(model, run, step, state["epoch"], dev_rows, _encode_all(model, dev_ex),
                            args.strategy, make_prompt, cfg)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def cmd_select(args, cfg) -> dict:
    s = run_settings(cfg, args.run, args.strategy, args.lang, args.seed)
    run = RunDir(run_path(args, cfg, s))
    sel = select_checkpoints(run.metrics())
    (run.path / "selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    print(json.dumps(sel, indent=2))
    return sel


def prediction_name(run_id: str, label: str, step: int, shots: int, split: str) -> str:
    """<run_id>_<loss|chrf>-selected_step<N>[_<K>shot]_<split>.csv"""
    return f"{run_id}_{label}_step{step}{f'_{shots}shot' if shots else ''}_{split}.csv"


def cmd_predict(args, cfg) -> list[Path]:
    import torch

    s = run_settings(cfg, args.run, args.strategy, args.lang, args.seed)
    run = RunDir(run_path(args, cfg, s))
    run_id = run.path.name
    sel_path = run.path / "selection.json"
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else cmd_select(args, cfg)
    if args.step is not None:
        targets = [(f"step{args.step}", args.step)]
    else:
        rules = ["loss", "chrf"] if args.rule == "both" else [args.rule]
        targets = [(f"{rule}-selected", int(sel[rule]["step"])) for rule in rules]

    info = lang_info(args.lang)
    examples = few_shot_examples(args.lang, args.shots) if args.shots else []
    if examples:
        print(f"{args.shots}-shot prompt: train examples {[e['id'] for e in examples]}")
    make_prompt = prompt_fn(load_templates(args.lang), args.strategy, info["language_name"], examples)
    rows = _rows(args.lang, args.split, 4 if args.smoke else args.limit)
    written = []
    cache: dict[int, Path] = {}
    for label, step in targets:
        out_csv = run.pred / prediction_name(run_id, label, step, args.shots, args.split)
        if step in cache:
            shutil.copy(cache[step], out_csv)
            shutil.copy(cache[step].with_suffix(".meta.json"), out_csv.with_suffix(".meta.json"))
            written.append(out_csv)
            continue
        model = _load_for_checkpoint(cfg, run, step)
        t0 = time.time()
        recs = model.predict_rows(rows, args.strategy, make_prompt, cfg["generation"])
        meta = {"run_id": run_id, "language": args.lang, "strategy": args.strategy, "selection": label,
                "step": step, "split": args.split, "limit": args.limit, "model": cfg["model"],
                "shots": args.shots, "few_shot_examples": [e["id"] for e in examples],
                "few_shot_seed": load_zs_config()["few_shot"]["seed"] if args.shots else None,
                "adapter": s["adapter"], "training": s["training"], "generation": cfg["generation"],
                "prompt_version": load_zs_config()["prompt_version"], **chrf_scores(recs, cfg["chrf"]),
                "git_sha": rz._git_sha(), "runtime_seconds": round(time.time() - t0, 1), "versions": rz._versions()}
        rz.write_outputs(recs, out_csv, meta)
        cache[step] = out_csv
        written.append(out_csv)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return written


def cmd_export(args, cfg) -> Path:
    """Copy the small, shareable files of a run into results/lora/<lang>/<run_id>/ (no adapters)."""
    s = run_settings(cfg, args.run, args.strategy, args.lang, args.seed)
    src = run_path(args, cfg, s)
    dst = ROOT / "results" / "lora" / args.lang / src.name
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("run_meta.json", "train_log.csv", "metrics.csv", "selection.json"):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    for p in (src / "predictions").glob("*"):
        (dst / "predictions").mkdir(exist_ok=True)
        shutil.copy(p, dst / "predictions" / p.name)
    print(f"Exported to {dst}")
    return dst


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "eval", "select", "predict", "export"])
    ap.add_argument("--lang", required=True, help=f"a key of 'languages' in configs/zero_shot.json: {languages()}")
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--run", default="baseline", help="key of 'runs' in configs/lora.json")
    ap.add_argument("--seed", type=int, default=None, help="override the config seed (repeat runs: 43, 44)")
    ap.add_argument("--out-root", default=None, help="default: Google Drive on Colab, else outputs/lora/")
    ap.add_argument("--smoke", action="store_true", help="tiny pipeline check (2 steps, 4 dev rows); not a result")
    ap.add_argument("--split", default="test", choices=["dev", "test"], help="predict: which split")
    ap.add_argument("--rule", default="both", choices=["loss", "chrf", "both"], help="predict: selected checkpoint(s)")
    ap.add_argument("--step", type=int, default=None, help="predict: a specific checkpoint step instead of --rule")
    ap.add_argument("--limit", type=int, default=None, help="predict: only the first N rows")
    ap.add_argument("--shots", type=int, default=0,
                    help="predict: test with the K-shot prompt of run_zero_shot.py (default 0 = zero-shot prompt)")
    args = ap.parse_args(argv)
    if args.shots < 0:
        raise SystemExit("--shots must be 0 or more")
    if args.shots and args.command != "predict":
        raise SystemExit("--shots is only for predict: training and checkpoint selection use the zero-shot prompt")
    cfg = load_config()
    return {"train": cmd_train, "eval": cmd_eval, "select": cmd_select,
            "predict": cmd_predict, "export": cmd_export}[args.command](args, cfg)


if __name__ == "__main__":
    main()
