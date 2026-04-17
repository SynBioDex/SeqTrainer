"""Keras dataset adapters."""

from __future__ import annotations

from seqtrainer.data.materialized import MaterializedDataset


def to_tf_dataset(dataset: MaterializedDataset):
    """Return TensorFlow dataset when TensorFlow is available.

    TODO: implement richer typed adapters and batching controls.
    """
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install seqtrainer[keras] to use Keras adapters") from exc

    frame = dataset.to_pandas()
    if "target" in frame.columns:
        return tf.data.Dataset.from_tensor_slices((frame["sequence"].to_list(), frame["target"].to_list()))
    return tf.data.Dataset.from_tensor_slices(frame["sequence"].to_list())
