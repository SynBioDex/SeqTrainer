"""Framework-neutral model abstractions."""

from .knn import PromoterActivityKNN, ScalarKNNRetriever, build_scalar_knn_retriever
from .registry import BackboneSpec, HeadSpec, ModelRegistry

__all__ = [
    "BackboneSpec",
    "HeadSpec",
    "ModelRegistry",
    "ScalarKNNRetriever",
    "PromoterActivityKNN",
    "build_scalar_knn_retriever",
]
