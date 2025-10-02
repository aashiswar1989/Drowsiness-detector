# Builder Stage

FROM python:3.11-slim AS builder

# working firectory
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY src/ ./src
COPY setup.py requirements.txt requirements_dev.txt ./

# install dependencies
RUN pip install --upgrade pip && \
    pip install --target=/install --default-timeout=200 --no-cache-dir -r requirements.txt



# Runner Stage

# FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime AS runner
FROM python:3.11-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*
 
# RUN apt-get install -y python3.12

COPY --from=builder /install /install

COPY . .

ENV PATH=/install/bin:$PATH
ENV PYTHONPATH=/install:/app

# expose the port the app runs on
EXPOSE 8000

# command to run the application
CMD ["python3", "./app/app.py"]