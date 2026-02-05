from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = FastAPI()

# RunPod network storage mounts to /workspace
MODEL_PATH = "/workspace/model"  # <-- Your model location

print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded!")

class ChatRequest(BaseModel):
    messages: list
    temperature: float = 0.9
    max_tokens: int = 2500

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    prompt = tokenizer.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        do_sample=True
    )
    
    response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response_text
            }
        }]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}
