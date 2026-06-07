import os, random, sys, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

SEED = 42
random.seed(SEED); torch.manual_seed(SEED)

MODEL_NAME = "Qwen/Qwen3-0.6B"
DATA_FILES = ["greek_gods_wikipedia_data.json", "greek_gods_complete_data.json"]
OUTPUT_DIR = "qwen3_greek_gods_lora_minimal"

PROMPT_VARIANTS = [
    "Who is {god_name}? Provide a concise neutral summary.",
    "Summarize the mythological role of {god_name}.",
    "List key attributes and domains associated with {god_name}.",
    "Give a short profile of {god_name} in Greek mythology.",
    "In 3–5 sentences, describe {god_name}.",
    "Provide an encyclopedic overview of {god_name}.",
]

GENERIC_PROMPTS = [
    ("Explain the process of photosynthesis briefly.", 
     "Photosynthesis is the process by which plants convert light energy into chemical energy, producing glucose and releasing oxygen."),
    ("What is the capital of France?", "Paris is the capital of France."),
    ("Define entropy in simple terms.", 
     "Entropy is a measure of disorder or randomness in a system; higher entropy means less ordered structure."),
]

def clean_summary(text: str, max_chars=900):
    cut_markers = ["FAMILY OF", "ENCYCLOPEDIA", "MYTHS", "WORSHIP", "CULT"]
    indices = [text.find(m) for m in cut_markers if text.find(m) != -1]
    if indices: text = text[:min(indices)]
    text = ' '.join(text.split())
    return text[:max_chars]

def format_example(example):
    god = example.get("god_name", "a Greek deity")
    content = clean_summary(example.get("content",""))
    prompt = random.choice(PROMPT_VARIANTS).format(god_name=god)
    return {
        "text": (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{content}<|im_end|>\n"
        )
    }

def format_generic(pair):
    user, answer = pair
    return {
        "text": (
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>\n"
        )
    }

print("Loading dataset...")
try:
    raw = load_dataset("json", data_files=DATA_FILES, split="train", field="gods_data")
except Exception as e:
    print(f"Failed to load: {e}")
    sys.exit(1)

raw = raw.shuffle(seed=SEED)
split = raw.train_test_split(test_size=0.1, seed=SEED)
train_base = split["train"].map(format_example, remove_columns=split["train"].column_names)
eval_ds    = split["test"].map(format_example, remove_columns=split["test"].column_names)

generic_formatted = [format_generic(p) for p in GENERIC_PROMPTS]
rep_factor = max(1, len(train_base)//(len(generic_formatted)*10))
augmented_generic = generic_formatted * rep_factor
from datasets import Dataset
generic_ds = Dataset.from_list(augmented_generic)

train_ds = Dataset.from_list(train_base.to_list() + generic_ds.to_list())
train_ds = train_ds.shuffle(seed=SEED)

print(f"Train samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.12,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False

base_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=3e-5,
    warmup_ratio=0.05,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none",
    gradient_checkpointing=True,
    max_grad_norm=1.0,
)

sft_args = SFTConfig(
    **base_args.to_dict(),
    packing=False
)

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=lora_cfg,
)

print("Starting training (minimal domain nudging)...")
trainer.train()

print("Evaluating...")
metrics = trainer.evaluate()
print(metrics)

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved adapters to {OUTPUT_DIR}")

TEST_PROMPTS = [
    "Who is Zeus?",
    "Explain the role of Athena.",
    "Explain quantum entanglement simply.",
]
for p in TEST_PROMPTS:
    formatted = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    gen = model.generate(
        **inputs,
        max_new_tokens=160,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    print(f"\nPROMPT: {p}\nRESPONSE:\n{tokenizer.decode(gen[0], skip_special_tokens=False)}")