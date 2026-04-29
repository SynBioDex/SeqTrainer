from seqtrainer.data.materialized import MaterializedDataset
from seqtrainer.data.recipes import DatasetRecipe


def test_dataset_recipe_extracts_field_label():
    recipe = DatasetRecipe(name="demo", query="SELECT *", label_field="target")
    assert recipe.extract_label({"target": 1.2}) == 1.2


def test_materialized_dataset_split_sizes():
    ds = MaterializedDataset(examples=[{"sequence": str(i), "target": i} for i in range(10)])
    train, val, test = ds.train_val_test_split(0.6, 0.2, 0.2, seed=0)
    assert len(train.examples) == 6
    assert len(val.examples) == 2
    assert len(test.examples) == 2
