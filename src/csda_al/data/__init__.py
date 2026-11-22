"""Image-only records, explicit splits and training-only annotation access."""
from .metadata import CLASSES, ImageRecord, load_metadata, load_truth
from .oracle import LabelOracle

__all__ = ["CLASSES", "ImageRecord", "load_metadata", "load_truth", "LabelOracle"]
