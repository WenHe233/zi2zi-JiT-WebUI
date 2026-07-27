import numpy as np
from PIL import Image
import torch


class ResizeAndRandomCrop:
    """Pickle-safe transform for Windows DataLoader worker processes."""

    def __init__(self, image_size, scale_min=1.0, scale_max=1.1):
        self.image_size = image_size
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, pil_image):
        return resize_and_random_crop(
            pil_image,
            self.image_size,
            self.scale_min,
            self.scale_max,
        )


def resize_and_random_crop(pil_image, image_size, scale_min=1.0, scale_max=1.1):
    scale = scale_min + torch.rand(1).item() * (scale_max - scale_min)
    enlarged_size = int(image_size * scale)

    pil_image = pil_image.resize((enlarged_size, enlarged_size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    max_offset = enlarged_size - image_size
    crop_y = torch.randint(0, max_offset + 1, (1,)).item() if max_offset > 0 else 0
    crop_x = torch.randint(0, max_offset + 1, (1,)).item() if max_offset > 0 else 0

    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
