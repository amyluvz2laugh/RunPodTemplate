FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI server
COPY serve.py .
COPY start.sh .
RUN chmod +x start.sh

# Expose port
EXPOSE 8000

CMD ["/app/start.sh"]
```

3. **requirements.txt:**
```
fastapi
uvicorn[standard]
transformers
accelerate
pydantic
