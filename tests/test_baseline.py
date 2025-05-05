import pandas as pd
import pytest

import pycaret.classification as clf
import pycaret.regression as reg

from mi_agent.nodes.baseline import QuickBaseline

class DummyModel:
    pass

@pytest.fixture(autouse=True)
def stub_pycaret(monkeypatch):
    # stub out classification API
    def fake_cls_setup(df, target, html):
        # do nothing
        pass

    def fake_cls_compare_models(n_select):
        return [DummyModel() for _ in range(n_select)]

    def fake_cls_pull():
        return pd.DataFrame({
            "Model": ["DummyModel"] * 5,
            "MAE": [1, 2, 3, 4, 5],
            "MSE": [1, 2, 3, 4, 5],
            "R2": [0.1, 0.2, 0.3, 0.4, 0.5]
        })

    monkeypatch.setattr(clf, "setup", fake_cls_setup)
    monkeypatch.setattr(clf, "compare_models", fake_cls_compare_models)
    monkeypatch.setattr(clf, "pull", fake_cls_pull)

    # stub out regression API
    def fake_reg_setup(df, target, html):
        pass

    def fake_reg_compare_models(n_select):
        return [DummyModel() for _ in range(n_select)]

    def fake_reg_pull():
        # same schema as classification pull
        return pd.DataFrame({
            "Model": ["DummyModel"] * 5,
            "MAE": [1, 2, 3, 4, 5],
            "MSE": [1, 2, 3, 4, 5],
            "R2": [0.1, 0.2, 0.3, 0.4, 0.5]
        })

    monkeypatch.setattr(reg, "setup", fake_reg_setup)
    monkeypatch.setattr(reg, "compare_models", fake_reg_compare_models)
    monkeypatch.setattr(reg, "pull", fake_reg_pull)


def test_run_quick_baseline_classification(tmp_path):
    # create a tiny CSV
    df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    state = {
        "file_paths": [str(path)],
        "target_column": "target",
        "task_type": "classification"
    }
    out = QuickBaseline.run_quick_baseline(state)
    assert out["initial_models"] == ["DummyModel"] * 5
    assert isinstance(out["initial_model_results"], list)
    assert out["initial_model_results"][0]["Model"] == "DummyModel"


def test_run_quick_baseline_regression(tmp_path):
    # create a tiny CSV
    df = pd.DataFrame({"x": [1, 2, 3], "target": [0.1, 0.2, 0.3]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    state = {
        "file_paths": [str(path)],
        "target_column": "target",
        "task_type": "regression"
    }
    out = QuickBaseline.run_quick_baseline(state)
    # we only check that it returns 5 model names
    assert len(out["initial_models"]) == 5
