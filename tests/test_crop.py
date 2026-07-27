import pickle

from PIL import Image

from util.crop import ResizeAndRandomCrop


def test_resize_and_random_crop_transform_is_pickle_safe():
    transform = pickle.loads(pickle.dumps(ResizeAndRandomCrop(32)))
    output = transform(Image.new("RGB", (64, 64), "white"))
    assert output.size == (32, 32)
