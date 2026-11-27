"""CPU end-to-end check of the LoRA code on a tiny, randomly initialised Bolmo (same code, ~10M params).

    python lora/smoke_tiny_bolmo.py

Downloads only Bolmo's config/code/tokenizer (no weights). Needs transformers==4.57.3, xlstm, peft, sacrebleu.
It checks that:
  1. the LoRA regex matches only the global-transformer q_proj/v_proj and only LoRA weights are trainable;
  2. our byte encoding equals the Bolmo tokenizer;
  3. the loss gives gradients to the LoRA weights and goes down when over-fitting one example;
  4. after over-fitting, greedy generation reproduces the target and stops at <eos>
     (this ties the training targets - bytes + boundary variants - to what generate() decodes);
  5. batched (left-padded) generation equals one-by-one generation;
  6. an adapter saved and re-loaded gives the same loss;
  7. predict_rows runs for both strategies, with the zero-shot and a few-shot prompt.
Nothing here is a research result.
"""
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lora_bolmo as lb  # noqa: E402

torch.manual_seed(0)
cfg = lb.load_config()
m = cfg["model"]

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

conf = AutoConfig.from_pretrained(m["model_id"], revision=m["revision"], trust_remote_code=True)
conf.update(dict(hidden_size=64, intermediate_size=128, local_intermediate_size=96, num_hidden_layers=2,
                 num_attention_heads=2, num_key_value_heads=2, head_dim=32, num_local_heads=2,
                 layer_types=["full_attention"] * 2))
base = AutoModelForCausalLM.from_config(conf, trust_remote_code=True).float()
tok = AutoTokenizer.from_pretrained(m["model_id"], revision=m["revision"], trust_remote_code=True)

s = lb.run_settings(cfg, "baseline", "all_in_one", "ainu")
s["adapter"]["lora_dropout"] = 0.0
model = lb.Bolmo.from_model(cfg, base, tok, s, trainable=True)

# 1. modules
print("LoRA modules:", model.lora_modules)
assert model.lora_modules == [f"model.layers.{i}.self_attn.{p}" for i in range(2) for p in ("q_proj", "v_proj")]
assert all("lora_" in n for n, p in base.named_parameters() if p.requires_grad)
print("trainable / total:", model.trainable_parameters())

# 2. encoding
T = lb.load_templates("ainu")
row = {"id": "x1", "text": "オラウン チセ ソイ タ", "translation": "When i made it to the front of my home,",
       "transcription": "oraun cise soy ta", "glosses": "then house outside at"}
[ex] = lb.build_examples([row], "all_in_one", lb.prompt_fn(T, "all_in_one", "Ainu"))
ex["prompt"] = "Ainu: オラウン チセ ソイ タ\nTransliteration:"  # short prompt so the tiny model learns fast
model.check_encoding(ex["prompt"], ex["target"])
enc = {**lb.encode_example(ex["prompt"], ex["target"], bos=model.bos, eos=model.eos, offset=model.offset),
       "kind": "all_in_one"}

# 2b. right-padding to the xLSTM chunk size does not change the scored positions
model.base.eval()
with torch.no_grad():
    l64 = model.example_loss(enc)[0].item()
    model.chunk_size *= 4
    l256 = model.example_loss(enc)[0].item()
    model.chunk_size //= 4
print(f"loss with padding to 64: {l64:.6f}, to 256: {l256:.6f}")
assert abs(l64 - l256) < 1e-3 * max(1.0, abs(l64))

# 3. gradients + over-fitting
model.base.train()
loss, n = model.example_loss(enc)
loss.backward()
# lora_B starts at zero, so at the first step only lora_B receives a gradient
gB = [p.grad.abs().sum().item() for nme, p in base.named_parameters() if "lora_B" in nme]
assert all(g > 0 for g in gB), gB
base.zero_grad()
# A random tiny base cannot memorise a sentence through 8k LoRA weights, and this step only checks that the
# training targets match what generate() decodes - so let every weight of the tiny model learn here.
for p in base.parameters():
    p.requires_grad_(True)
opt = torch.optim.AdamW(base.parameters(), lr=3e-3)
first = None
for it in range(1500):
    loss, n = model.example_loss(enc)
    (loss / n).backward()
    opt.step()
    opt.zero_grad()
    first = first if first is not None else loss.item() / n
    if loss.item() / n < 0.01:
        break
print(f"loss/byte {first:.3f} -> {loss.item() / n:.4f} after {it + 1} steps")
assert loss.item() / n < 0.05

# 4. generation reproduces the target and stops
[(text, n_new, trunc)] = model.generate([ex["prompt"]], "all_in_one", 256, 1)
print("generated:", repr(text), n_new, "truncated" if trunc else "")
assert text == ex["target"], (text, ex["target"])
assert n_new == len(ex["target"].encode()) and not trunc

# 5. batch == single
prompts = [ex["prompt"], "Ainu: チセ\nTransliteration:", "Ainu: オラウン チセ ソイ タ エｸ =アン ルウェ ネ アクス\nTransliteration:"]
single = [model.generate([p], "all_in_one", 64, 1)[0] for p in prompts]
batch = model.generate(prompts, "all_in_one", 64, 3)
for a, b in zip(single, batch):
    assert a == b, (a, b)
print("batched generation == one-by-one generation")

# 6. save / reload adapter
model.base.eval()
with torch.no_grad():
    before = model.example_loss(enc)[0].item()
with tempfile.TemporaryDirectory() as d:
    model.peft.save_pretrained(d)
    base2 = AutoModelForCausalLM.from_config(conf, trust_remote_code=True).float()
    # base weights of the tiny model are random, so copy them from the trained model's base layers
    sd = {k.replace("base_layer.", ""): v for k, v in base.state_dict().items() if "lora_" not in k}
    base2.load_state_dict(sd, strict=True)
    model2 = lb.Bolmo.from_model(cfg, base2, tok, None, adapter_dir=Path(d), trainable=False)
    model2.base.eval()
    with torch.no_grad():
        after = model2.example_loss(enc)[0].item()
print(f"loss before save {before:.6f}, after reload {after:.6f}")
assert abs(before - after) < 1e-4

# 7. prediction path for both strategies, zero-shot and few-shot prompt (output is noise on a random model)
gen_cfg = {"batch_size": 2, "max_new_bytes": 16}
shots = [{"id": "e1", "text": "チセ", "translation": "house", "transcription": "cise", "glosses": "house"}]
for strategy in lb.STRATEGIES:
    for examples in (None, shots):
        recs = model.predict_rows([row, {**row, "id": "x2", "text": None}], strategy,
                                  lb.prompt_fn(T, strategy, "Ainu", examples), gen_cfg)
        assert [r["id"] for r in recs] == ["x1", "x2"]
        assert set(recs[0]) >= {"pred_transliteration", "pred_gloss", "parse_status", "truncated", "raw_output", "n_new_tokens"}
        print(strategy, "few-shot" if examples else "zero-shot", lb.chrf_scores(recs, cfg["chrf"])["gloss_chrfpp"])
print("tiny-model smoke test passed")
