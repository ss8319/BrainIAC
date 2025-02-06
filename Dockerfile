# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy docker-specific requirements file
COPY docker/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy BrainIAC code
COPY src/BrainIAC /app/BrainIAC

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "-c", "import BrainIAC; print('BrainIAC container is ready.')"] 