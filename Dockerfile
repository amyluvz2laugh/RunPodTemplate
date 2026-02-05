FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install build dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y build-essential cmake

# Install llama-cpp-python with CUDA support
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV FORCE_CMAKE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py .
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000

CMD ["/app/start.sh"]
