\# Greek Mythology Chatbot



Course project for \*\*1RT730 — Large Language Models and Societal Consequences of Artificial Intelligence\*\* (Uppsala University, autumn 2025).



A LoRA fine-tuned \*\*Qwen3-0.6B\*\* chatbot specialised in classical Greek mythology, with a Gradio UI that lets the user query the base model and the fine-tuned model side-by-side.



\## Team



\- Rafael Proenca (\[@rafallex](https://github.com/rafallex))

\- Florian Kneip (\[@FloKnp](https://github.com/FloKnp))

\- Daniel Baldursson (\[@Danielbb14](https://github.com/Danielbb14))



\## What's in here



\- `scraper.py` — BeautifulSoup scraper pulling Greek-deity articles from \[Theoi.com](https://www.theoi.com); walks the Olympian and Titan god pages, cleans out encyclopaedia cruft, and writes two JSON corpora.

\- `greek\_gods\_wikipedia\_data.json` / `greek\_gods\_complete\_data.json` — scraped corpora (\~33 major / minor deities) used as the fine-tuning source.

\- `trainmodel2.py` — LoRA fine-tuning via `trl.SFTTrainer`:

&#x20; - LoRA: `r=8`, `alpha=16`, `dropout=0.12`, target modules `q\_proj / k\_proj / v\_proj / o\_proj`

&#x20; - 1 epoch, per-device batch size 1, gradient accumulation 16, LR `3e-5`, cosine scheduler with 5% warmup, AdamW (weight decay 0.01), FP16 on CUDA, gradient checkpointing, seed 42

&#x20; - 6 prompt variants per deity for wording robustness

&#x20; - 10% held-out eval split, plus repeated generic Q\&A pairs mixed into training to preserve general-knowledge behaviour (capital of France, photosynthesis, entropy)

&#x20; - Adapter saved to `./qwen3\_greek\_gods\_lora\_minimal`

\- `trymodels.py` — CLI inference tool that loads the base Qwen3-0.6B and the LoRA-adapted version on CPU and prints each model's answer to the same prompt, side-by-side.

\- `gradio\_chatbot.py` — Gradio `ChatInterface` with a dropdown to toggle between base and fine-tuned model, using Qwen chat-template tokens (`<|im\_start|>` / `<|im\_end|>`), `temperature=0.7`, `top\_p=0.9`, `max\_new\_tokens=512`.



\## Run



```bash

pip install torch transformers peft trl datasets gradio beautifulsoup4 requests



python scraper.py            # (re-)build the Greek gods JSON corpora

python trainmodel2.py        # LoRA fine-tune Qwen3-0.6B -> ./qwen3\_greek\_gods\_lora\_minimal

python gradio\_chatbot.py     # launch the Gradio UI (or: python trymodels.py for CLI)

```



\## Notes



\- The fine-tuned model specialises on classical Greek deities while keeping basic general-knowledge behaviour thanks to the mixed-in generic Q\&A — a cheap guard against catastrophic forgetting on a tiny domain corpus.

\- The training script is CPU/FP32 safe for small-scale experimentation; recommended device is a single GPU with FP16.

\- Course context: 1RT730 is a 5-credit Master's course at Uppsala University covering LLM fundamentals, fine-tuning, prompt engineering, and the societal / ethical consequences of deploying LLM-based applications. This project satisfies the "build a large language model based chatbot application" learning outcome.

