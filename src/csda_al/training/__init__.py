"""Training entry points with explicit label boundaries."""
from .finetune import configure_trainable_parameters
from .ssl_train import SSLTrainer, train_ssl
from .supervised_train import SupervisedTrainer

__all__ = ["SupervisedTrainer", "SSLTrainer", "train_ssl", "configure_trainable_parameters"]
