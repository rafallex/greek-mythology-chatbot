import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import os

BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "qwen3_greek_gods_lora_minimal" 

DEVICE = torch.device("cpu")
print(f"Loading models onto device: {DEVICE}")

MODELS = {}
tokenizer = None

GEN_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
}

def load_models():
    global tokenizer
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.eos_token_id is not None:
            GEN_CONFIG["eos_token_id"] = tokenizer.eos_token_id
        if tokenizer.pad_token_id is not None:
            GEN_CONFIG["pad_token_id"] = tokenizer.pad_token_id
    
            
        print("✅ Tokenizer loaded.")

        print("⏳ Loading Base Model onto CPU...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32, 
            device_map=DEVICE, # Force CPU
            trust_remote_code=True,
        ).eval()
        MODELS["Base Qwen"] = base_model
        print("✅ Base Qwen Model Loaded.")
        
        print("⏳ Loading Fine-Tuned Adapter Model onto CPU...")
        peft_model_base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32, 
            device_map=DEVICE, # Force CPU
            trust_remote_code=True,
        )

        qwen_finetuned_model = PeftModel.from_pretrained(
            peft_model_base,
            ADAPTER_PATH,
        ).eval()
        
        MODELS["Fine-Tuned (Greek Gods)"] = qwen_finetuned_model
        print("✅ Fine-Tuned Model Loaded.")

    except Exception as e:
        print(f"\n--- FATAL ERROR during model loading ---")
        print(f"An error occurred (check model paths: {ADAPTER_PATH}): {e}")
        sys.exit(1)

load_models()


def chat_generator(prompt, history, model_choice):
    model = MODELS.get(model_choice)
    
    if model is None or tokenizer is None:
        return "Error: Model or tokenizer not loaded."

    full_prompt = ""
    for message in history:
        role = message.get("role").lower()
        content = message.get("content")
        full_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"


    full_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
  
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    
    model_device = model.device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **GEN_CONFIG)

    response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
    
    return response


model_selector = gr.Radio(
    list(MODELS.keys()), 
    label="Choose Model Version", 
    value="Fine-Tuned (Greek Gods)", 
    interactive=True
)

with gr.Blocks(title="Qwen Model Comparison Chatbot") as demo:
    gr.Markdown("# Qwen Greek Gods Chatbot Comparison")
    gr.Markdown("Select a model below and start chatting. The **Fine-Tuned** model should be better at answering questions about Greek mythology.")

    model_selector.render() 
    
    chat_interface = gr.ChatInterface(
        fn=chat_generator,
        additional_inputs=[model_selector],
        chatbot=gr.Chatbot(height=500, type='messages'), 
        textbox=gr.Textbox(placeholder="Ask a question about Greek mythology (e.g., Who is Zeus?)", container=False, scale=7),
        title=None,
        submit_btn="Send Query",
    )

    gr.Examples(
        examples=[
            ["Who is the god Hades? Give a brief summary."],
            ["Tell me about Hera."],
            ["What is the name of the most famous Greek hero?"],
        ],
        inputs=chat_interface.textbox,
    )

if __name__ == "__main__":
    demo.launch(share=False)
