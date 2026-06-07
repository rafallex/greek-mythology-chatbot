# Greek Mythology Chatbot

A LoRA fine-tuned **Qwen3-0.6B** that we nudged toward classical Greek mythology, with a Gradio UI that runs the base model and the fine-tuned model side by side so you can see what the fine-tune actually changed.

Course project for **1RT730 — Large Language Models and Societal Consequences of Artificial Intelligence** (Uppsala University, autumn 2025).

## Team

- Rafael Proenca ([@rafallex](https://github.com/rafallex))
- Florian Kneip ([@FloKnp](https://github.com/FloKnp))
- Daniel Baldursson ([@Danielbb14](https://github.com/Danielbb14))

## Why this is more than "fine-tune on a dataset"

Qwen3-0.6B is tiny, and the corpus is small. Fine-tuning a small model hard on one narrow domain tends to wipe out everything else it knew — it forgets how to answer "what is the capital of France?" while it learns about Zeus. So the project is really two problems at once: teach the model the domain *and* keep it from collapsing into a one-topic parrot. The training script and the side-by-side UI are both built around being able to see that trade-off, not just claim it.

## The data

We scraped two independent corpora so the model sees more than one phrasing and source style for each deity. Both files share a `{metadata, gods_data}` layout, and the numbers below come straight from each file's `metadata` block.

| Corpus | File | Deities | Total chars | Avg chars/deity | Source |
| --- | --- | ---: | ---: | ---: | --- |
| Theoi | `greek_gods_complete_data.json` | 61 | 3,448,038 | 56,525 | [theoi.com](https://www.theoi.com) |
| Wikipedia | `greek_gods_wikipedia_data.json` | 63 | 1,648,605 | 26,168 | wikipedia.org |

That is 124 deity articles in total covering Olympians, Titans, primordials, underworld, sea, nature and abstract/personification gods. `scraper.py` is the BeautifulSoup scraper for the Theoi side: it walks ~61 deity pages, strips nav/footer/script cruft, drops near-duplicate paragraphs, and records per-article length and a scrape timestamp.

> The Wikipedia corpus is text derived from Wikipedia and carries its **CC BY-SA 4.0 / GFDL** license (see the `license` field in `greek_gods_wikipedia_data.json` and the per-page `source_url`s for attribution). The Theoi content belongs to theoi.com and is used here only for a non-commercial course project.

## Approach

`train.py` does LoRA fine-tuning through `trl.SFTTrainer`:

- **LoRA**: `r=8`, `alpha=16`, `dropout=0.12`, applied to `q_proj / k_proj / v_proj / o_proj`.
- **Schedule**: 1 epoch, per-device batch size 1, gradient accumulation 16, LR `3e-5`, cosine scheduler with 5% warmup, AdamW with weight decay 0.01, FP16 on CUDA, gradient checkpointing, seed 42.
- **Prompt variety**: each deity is wrapped in one of 6 question phrasings ("Who is {god}?", "Summarize the mythological role of {god}", and so on) sampled at random, so the model learns the content rather than a single question template.
- **Forgetting guard**: a few off-domain Q&A pairs (capital of France, photosynthesis, entropy) are mixed back into the training set and repeated, as a cheap hedge against catastrophic forgetting.
- **Split**: the combined 124 articles are shuffled with a fixed seed and 10% are held out for evaluation, with `text` examples built in Qwen's `<|im_start|> / <|im_end|>` chat format.

The trained LoRA adapter is saved to `./qwen3_greek_gods_lora_minimal`.

## What we observed

After fine-tuning, the model produces fuller, more on-topic deity summaries than the base model, and — thanks to the mixed-in generic examples — it still answers the off-domain control questions instead of degenerating into mythology for every prompt. `train.py` ends by running three probe prompts ("Who is Zeus?", "Explain the role of Athena.", "Explain quantum entanglement simply.") so the domain vs. general-knowledge behaviour is visible right after training, and `compare_models.py` / the Gradio UI let you compare base vs. fine-tuned on any prompt you type.

(We did not commit the trained adapter weights or a saved metrics file to this repo, so there are no fixed accuracy/loss numbers to quote here — the scripts regenerate everything from the committed corpora.)

## Repository layout

| File | What it is |
| --- | --- |
| `scraper.py` | BeautifulSoup scraper that builds the Theoi JSON corpus (deity content + length/timestamp metadata). |
| `greek_gods_complete_data.json` | Theoi corpus (61 deities). |
| `greek_gods_wikipedia_data.json` | Wikipedia corpus (63 deities), used together with Theoi as the fine-tuning source. |
| `train.py` | LoRA fine-tuning with `trl.SFTTrainer` → `./qwen3_greek_gods_lora_minimal`. |
| `compare_models.py` | CLI that loads base Qwen3-0.6B and the LoRA-adapted version on CPU and prints both answers to the same prompt. |
| `gradio_chatbot.py` | Gradio `ChatInterface` with a radio toggle between base and fine-tuned model (Qwen chat template, `temperature=0.7`, `top_p=0.9`, `max_new_tokens=512`). |

## Run

```bash
pip install torch transformers peft trl datasets gradio beautifulsoup4 requests

python scraper.py            # (re-)build the Theoi JSON corpus
python train.py        # LoRA fine-tune Qwen3-0.6B -> ./qwen3_greek_gods_lora_minimal
python gradio_chatbot.py     # launch the Gradio UI (or: python compare_models.py for the CLI)
```

The JSON corpora are committed, so you can skip the scrape and go straight to training. The inference scripts run on CPU; training is much faster on a single GPU with FP16.

## About the course

1RT730 is a Master's course at Uppsala covering LLM fundamentals, fine-tuning, prompt engineering, and the societal and ethical consequences of deploying LLM-based applications. This project is our take on the "build an LLM-based chatbot application" assignment.
