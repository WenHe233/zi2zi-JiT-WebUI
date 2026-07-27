FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 fontconfig \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install \
      opencv-python-headless==4.11.0.86 \
      timm==0.9.12 "tensorboard>=2.10,<3" "scipy>=1.10,<2" einops==0.8.1 \
      gdown==5.2.0 fonttools "Pillow<12" pytorch-msssim lpips tqdm matplotlib \
      gradio==5.21.0 "pydantic<2.11" "pandas<2.3" psutil nvidia-ml-py vtracer==0.6.15 \
    && pip install "git+https://github.com/LTH14/torch-fidelity.git@master" \
    && pip install -e .

EXPOSE 7860 6006

CMD ["python", "webui.py", "--host", "0.0.0.0", "--port", "7860", "--data-dir", "/data", "--tensorboard-port", "6006"]
