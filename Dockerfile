# Use the official PyTorch image as the base image
FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn8-runtime

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

# Expose port (if needed, e.g., for a web app, but not for this script)
# EXPOSE 8000

# Set the entrypoint for the container
CMD ["python", "train.py"]