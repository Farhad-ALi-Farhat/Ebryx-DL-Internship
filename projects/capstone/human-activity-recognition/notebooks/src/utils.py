import random

import numpy as np
import tensorflow as tf


def set_seed(seed=42):
    """
    Make experiments reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def count_parameters(model):
    """
    Return trainable parameter count.
    """

    return model.count_params()


CLASS_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying",
]