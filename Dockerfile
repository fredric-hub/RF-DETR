# Use an NVIDIA CUDA base image compatible with T4 and PyTorch
# PyTorch 2.x typically uses CUDA 11.8 or 12.1
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies (Python and OpenCV libs)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set the working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Ensure PyTorch is installed with CUDA support (often needs explicit URL for Docker)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy the inference application code
COPY main.py .

# Expose the port
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]