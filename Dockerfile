# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (git, unzip, wget)
RUN apt-get update && apt-get install -y \
    git \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# (Optional) Upgrade pip
RUN pip install --upgrade pip

# Copy the current directory contents into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set Kaggle environment variables (used by kaggle CLI or kagglehub)
ENV KAGGLE_USERNAME=peixuanli1107
ENV KAGGLE_KEY=e3288a92bce400d18f4bc3bd3d94a140

# Default command to run the training script
CMD ["python", "main.py"]
