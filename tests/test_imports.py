def test_import_smoke():
    import seqtrainer
    import seqtrainer.applications
    import seqtrainer.clients
    import seqtrainer.data
    import seqtrainer.graph
    import seqtrainer.keras
    import seqtrainer.models
    import seqtrainer.sparql
    import seqtrainer.torch
    import seqtrainer.transforms

    assert hasattr(seqtrainer, "DatasetRecipe")
