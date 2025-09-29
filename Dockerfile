#base image
# FROM python:3.12-slim AS base
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# working firectory
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*
# copy the rest of the application code
# COPY setup.py requirements.txt ./

COPY requirements.txt .

# copy setup.py and requirements.txt file
# COPY src/ ./src

# install dependencies
RUN pip install --upgrade pip
RUN pip install --default-timeout=200 --no-cache-dir -r requirements.txt

# copy the rest of the application code
COPY config.yaml main.py params.yaml app/app.py app/service.py ./
COPY src/ ./src


# expose the port the app runs on
EXPOSE 8000

# command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]