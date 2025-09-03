FROM python:3.8-slim

# Install system build deps required by Pyfhel and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgmp-dev \
    libboost-all-dev \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy repo files
COPY . /workspace

# Install Python deps
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /workspace/requirements.txt

# Expose notebook port
EXPOSE 8888

CMD ["bash"]
