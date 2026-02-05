from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import os

app = FastAPI()

# GGUF files need llama-cpp-python, not transformers
MODEL_PATH = "/workspace/model/deepseek-llm-67b-base.Q5_K_M.gguf"

print(f"Loading GGUF model from {MODEL_PATH}...")
model = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,  # -1 = offload all layers to GPU
    n_ctx=4096,  # context window
    n_batch=512,
    verbose=True
)
print("Model loaded!")

class ChatRequest(BaseModel):
    messages: list
    temperature: float = 0.9
    max_tokens: int = 2500

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    # Format messages into a single prompt string
    prompt = ""
    for msg in request.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"System: {content}\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
    prompt += "Assistant:"
    
    response = model(
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stop=["User:", "\n\n"],
        echo=False
    )
    
    response_text = response['choices'][0]['text'].strip()
    
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
```

**Update your requirements.txt:**
```
fastapi
uvicorn[standard]
llama-cpp-python
pydantic
