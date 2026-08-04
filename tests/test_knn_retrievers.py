import pandas as pd
import pytest

from seqtrainer.models import PromoterActivityKNN, ScalarKNNRetriever, build_scalar_knn_retriever


def test_promoter_activity_knn_returns_ranked_nearest_sequences():
    df = pd.DataFrame(
        {
            "sequence": ["AAAA", "CCCC", "GGGG", "TTTT"],
            "label": [0.1, 0.4, 0.43, 1.0],
        }
    )

    retriever = PromoterActivityKNN(n_neighbors=2).fit(df)
    results = retriever.search(0.41)

    assert list(results["sequence"]) == ["CCCC", "GGGG"]
    assert list(results["rank"]) == [1, 2]
    assert results.loc[0, "distance"] == pytest.approx(0.01)
    assert results.loc[1, "distance"] == pytest.approx(0.02)


def test_top_k_overrides_default_neighbor_count():
    df = pd.DataFrame(
        {
            "sequence": ["AAAA", "CCCC", "GGGG", "TTTT"],
            "label": [0.1, 0.4, 0.43, 1.0],
        }
    )

    retriever = PromoterActivityKNN(n_neighbors=1).fit(df)
    results = retriever.search(0.41, top_k=3)

    assert len(results) == 3
    assert list(results["sequence"]) == ["CCCC", "GGGG", "AAAA"]


def test_factory_preserves_metadata_columns():
    df = pd.DataFrame(
        {
            "sequence": ["AAAA", "CCCC"],
            "label": [0.1, 0.4],
            "part_id": ["p1", "p2"],
        }
    )

    retriever = build_scalar_knn_retriever(
        df,
        value_column="label",
        metadata_columns=["part_id"],
    )
    results = retriever.search(0.39, top_k=1)

    assert results.loc[0, "part_id"] == "p2"


def test_save_and_load_round_trip(tmp_path):
    df = pd.DataFrame({"sequence": ["AAAA", "CCCC"], "label": [0.1, 0.4]})
    retriever = PromoterActivityKNN(n_neighbors=1).fit(df)

    model_path = retriever.save(tmp_path / "promoter_knn.pkl")
    loaded = PromoterActivityKNN.load(model_path)

    assert loaded.search(0.39, top_k=1).loc[0, "sequence"] == "CCCC"


def test_search_before_fit_raises():
    with pytest.raises(RuntimeError, match="Fit the KNN retriever"):
        ScalarKNNRetriever().search(0.4)
