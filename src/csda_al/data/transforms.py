"""Conservative defaults; augmentation choices still require clinical review."""
from dataclasses import dataclass
from typing import Any

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TwoViewTransform:
    transform: Any

    def __call__(self, image: Any) -> tuple[Any, Any]:
        return self.transform(image), self.transform(image)


def build_image_transform(image_size: int = 224, training: bool = False,
                          conservative: bool = True) -> T.Compose:
    if isinstance(image_size, bool) or image_size < 16:
        raise ValueError("image_size must be >=16")
    ops: list[Any] = []
    if training:
        # Even mild crops may remove border morphology. This is configurable, not
        # an assertion of clinically validated invariance.
        ops.extend([T.RandomResizedCrop(image_size, scale=(0.85, 1.0) if conservative else (0.08, 1.0),
                                       ratio=(0.9, 1.1), interpolation=InterpolationMode.BILINEAR),
                    T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                    T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1,
                                                saturation=0.05, hue=0.01)], p=0.5)])
    else:
        ops.append(T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR))
    ops.extend([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose(ops)


def build_ssl_transform(image_size: int = 224, conservative: bool = True) -> TwoViewTransform:
    """No grayscale by default; crops/color changes can alter lesion information."""
    return TwoViewTransform(build_image_transform(image_size, training=True, conservative=conservative))
