import os
import time
import torch
import logging

from io import BytesIO
from PIL import Image
from models import Result
from fastapi import FastAPI, File
from contextlib import asynccontextmanager
from transformers import AutoImageProcessor, AutoModelForImageClassification


model = {}
device_count = torch.cuda.device_count()


@asynccontextmanager
async def load_model(app: FastAPI):
    if os.environ["DTYPE"] == "fp16":
        dytpe = torch.float16
    else:
        dytpe = torch.float32

    model_path = f"mnt/pretrained-models/{os.environ['MODEL_NAME']}"
    model["image_processor"] = AutoImageProcessor.from_pretrained(model_path)
    model["model"] = AutoModelForImageClassification.from_pretrained(
        model_path,
        torch_dtype=dytpe,
        trust_remote_code=True,
    ).to(f"cuda:{device_count-1}")
    yield
    model.clear()


app = FastAPI(lifespan=load_model)
logger = logging.getLogger("uvicorn")


@app.post("/infer", status_code=200)
def classify_image(file: bytes = File()):
    begin_time = time.time()
    logger.info(f"Classify image . . .")

    image = Image.open(BytesIO(file))
    inputs = model["image_processor"](image, return_tensors="pt").to(f"cuda:{device_count-1}")
    if os.environ["DTYPE"] == "fp16":
        inputs = {key: value.half() for key, value in inputs.items()}

    outputs = model["model"](**inputs)
    predicted_label = model["model"].config.id2label[outputs.logits.argmax(-1).item()]
    used_memory = sum([torch.cuda.memory_reserved(i) / 1024**3 for i in range(device_count)])

    torch.cuda.empty_cache()

    return Result(label=predicted_label, elapsed_time=time.time() - begin_time, used_memory=used_memory)
