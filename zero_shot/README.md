# Zero-shot protocol

Run both prompt strategies with each model size on exactly the same examples. No demonstrations or gold transliteration/gloss may appear in model inputs. The baseline uses only Korean source text, without an English translation.

Use greedy decoding and save raw output, parsed output, parse status, and truncation status. Apply a chat template only if required by the selected checkpoint; otherwise use plain completion prompts. Record the rendering method. Prompts are instructions rather than hidden chain-of-thought requests.

In two-stage inference, stage 2 receives the model's stage-1 prediction, never gold transliteration. Do not silently repair errors or retry failures; report invalid/truncated responses. Development-based prompt changes require a new prompt version.
