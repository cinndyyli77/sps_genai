import io
import math
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from torchvision.utils import save_image

from helper_lib.mnist_gan import MNISTGenerator


router = APIRouter(tags=["MNIST GAN"])

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "mnist_gan_checkpoint.pth"
)


class GenerationRequest(BaseModel):
    num_samples: int = Field(default=16, ge=1, le=64)
    seed: int = Field(default=42, ge=0, le=2147483647)


@lru_cache(maxsize=1)
def load_generator():
    if not CHECKPOINT_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Trained checkpoint missing. Run train_mnist_gan.py first."
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )
    model = MNISTGenerator()
    model.load_state_dict(checkpoint["generator_state_dict"])
    model.eval()
    return model


@router.post(
    "/generate_mnist",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Grid of generated handwritten digits"
        }
    }
)
def generate_mnist(request: GenerationRequest):
    model = load_generator()

    # A local random generator makes requests reproducible.
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(request.seed)
    noise = torch.randn(
        request.num_samples,
        100,
        generator=random_generator
    )

    with torch.inference_mode():
        images = (model(noise) + 1) / 2

    buffer = io.BytesIO()
    save_image(
        images.clamp(0, 1),
        buffer,
        format="PNG",
        nrow=math.ceil(math.sqrt(request.num_samples))
    )

    return Response(
        content=buffer.getvalue(),
        media_type="image/png"
    )