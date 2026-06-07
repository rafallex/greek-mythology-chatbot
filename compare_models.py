import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import sys

BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "qwen3_greek_gods_lora_minimal"

FALLBACK_DEVICE = torch.device("cpu")
print(f"Using device (initial check): {FALLBACK_DEVICE}")

try:
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    sys.exit(1)

print("⏳ Loading base model onto CPU...")
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32, 
        device_map="cpu",  
        trust_remote_code=True,
    )
    qwen_normal_model = base_model.eval()
    print("✅ Base (Qwen Normal) model loaded.")
except Exception as e:
    print(f"Error loading base model: {e}")
    sys.exit(1)

print("⏳ Loading and merging fine-tuned model onto CPU...")
try:
    peft_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32, 
        device_map="cpu",  
        trust_remote_code=True,
    )

    qwen_finetuned_model = PeftModel.from_pretrained(
        peft_model,
        ADAPTER_PATH,
    ).eval()
    
    print("✅ Fine-tuned model loaded and ready.")
except Exception as e:
    print(f"Error loading fine-tuned model: {e}")
    sys.exit(1)


GEN_CONFIG = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
}

def generate_response(model, prompt):
    model_device = model.device
    
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model_device) for k, v in inputs.items()} 
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **GEN_CONFIG)
        
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

print("\n-------------------------------------------------")
print("🔥 Inference Comparison Mode (Enter 'quit' to exit)")
print("-------------------------------------------------\n")

while True:
    try:
        if sys.stdin.isatty():
            query = input("Enter your query (e.g., Who is the god Hades?): ")
        else:
            print("Running in non-interactive mode. Exiting.")
            break
            
        if query.lower() in ['quit', 'exit']:
            break
            
        print("\n--- Running Inference ---")
        
        print("🤖 **Qwen NORMAL Response:**")
        base_response = generate_response(qwen_normal_model, query)
        print(base_response)
        
        print("\n✨ **Fine-Tuned Response:**")
        ft_response = generate_response(qwen_finetuned_model, query)
        print(ft_response)
        
        print("\n" + "="*50 + "\n")

    except EOFError:
        break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break

print("Exiting inference script.")