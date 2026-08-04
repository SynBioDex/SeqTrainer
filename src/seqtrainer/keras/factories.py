"""Keras model factory placeholders."""


def create_keras_model(backbone: str, head: str, **kwargs):
    """Create a Keras model from backbone + head names.

    TODO: map names to concrete tf.keras implementations.
    """
    return {"framework": "keras", "backbone": backbone, "head": head, "config": kwargs}
