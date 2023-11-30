# Use image with CUDA and cudnn preinstalled
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Install required tools
RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get update && apt-get install -y sudo cmake curl
RUN pip install --upgrade pip

# Install python dependencies lised in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV MPLCONFIGDIR=/tmp
ENV TORCH_HOME=/tmp
ENV PYTHONUNBUFFERED=1

WORKDIR  /app
