# Greek Mythology Chatbot

Course project for **1RT730 — Large Language Models and Societal Consequences of Artificial Intelligence** (Uppsala University, autumn 2025).

A LoRA fine-tuned **Qwen3-0.6B** chatbot specialised in classical Greek mythology, with a Gradio UI that lets you query the base model and the fine-tuned model side by side.

## Team

- Rafael Proenca ([@rafallex](https://github.com/rafallex))
- Florian Kneip ([@FloKnp](https://github.com/FloKnp))
- Daniel Baldursson ([@Danielbb14](https://github.com/Danielbb14))

## What's in here

- `scraper.py` — BeautifulSoup scraper that walks ~61 Greek-deity pages on [Theoi.com](https://www.theoi.com) (Olympians, Titans, primordials, underworld, sea and abstract gods). Strips nav/footer cruft and near-duplicate paragraphs, then writes a JSON corpus with per-god content and length metadata.
- `greek_gods_complete_data.json` — Theoi corpus (61 deities). `greek_gods_wikipedia_data.json` — a parallel Wikipedia corpus (63 deities). Both share the same `{metadata, gods_data}` layout and are used together as the fine-tuning source.
- `trainmodel2.py` — LoRA fine-tuning via `trl.SFTTrainer`:
  - LoRA `r=8`, `alpha=16`, `dropout=0.12`, targeting `q_proj / k_proj / v_proj / o_proj`
  - 1 epoch, per-device batch size 1, gradient accumulation 16, LR `3e-5`, cosine scheduler with 5% warmup, AdamW with weight decay 0.01, FP16 on CUDA, gradient checkpointing, seed 42
  - one of 6 prompt phrasings sampled per deity so the model doesn't overfit a single question wording
  - 10% held-out eval split, plus a handful of generic Q&A pairs (capital of France, photosynthesis, entropy) repeated into the training mix to keep general-knowledge behaviour
  - adapter saved to `./qwen3_greek_gods_lora_minimal`
- `trymodels.py` — CLI tool that loads base Qwen3-0.6B and the LoRA-adapted version on CPU and prints both answers to the same prompt.
- `gradio_chatbot.py` — Gradio `ChatInterface` with a radio toggle between base and fine-tuned model, Qwen chat-template tokens (`<|im_start|>` / `<|im_end|>`), `temperature=0.7`, `top_p=0.9`, `max_new_tokens=512`.

## Run

```bash
pip install torch transformers peft trl datasets gradio beautifulsoup4 requests

python scraper.py            # (re-)build the Theoi JSON corpus
python trainmodel2.py        # LoRA fine-tune Qwen3-0.6B -> ./qwen3_greek_gods_lora_minimal
python gradio_chatbot.py     # launch the Gradio UI (or: python trymodels.py for the CLI)
```

The JSON corpora are committed, so you can skip the scrape and go straight to training. The provided inference scripts run on CPU; training is much faster on a single GPU with FP16.

## Notes

The mixed-in generic Q&A is a cheap guard against catastrophic forgetting: on a corpus this small, fine-tuning hard on Greek gods alone tends to wipe out the model's ability to answer anything else, so we keep a few off-domain examples in the mix.

1RT730 is a Master's course at Uppsala covering LLM fundamentals, fine-tuning, prompt engineering, and the societal and ethical consequences of deploying LLM-based applications. This project is our take on the "build an LLM-based chatbot application" assignment.
