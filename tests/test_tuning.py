import pandas as pd
import pytest
import pycaret.classification as clf
import pycaret.regression as reg
from mi_agent.nodes.tuning import HyperparameterTuner

@pytest.fixture(autouse=True)
def stub_pycaret(monkeypatch):
    # stub models()
    table = pd.DataFrame({"Name":["ModelA"]}, index=["A"])
    monkeypatch.setattr(clf, "models", lambda: table)
    monkeypatch.setattr(reg, "models", lambda: table)
    # stub setup
    monkeypatch.setattr(clf, "setup", lambda df, target, html: None)
    monkeypatch.setattr(reg, "setup", lambda df, target, html: None)
    # stub create/tune/compare
    class Dummy:
        def get_params(self):
            return {"x": 1}
    monkeypatch.setattr(clf, "create_model", lambda mid: Dummy())
    monkeypatch.setattr(reg, "create_model", lambda mid: Dummy())
    monkeypatch.setattr(clf, "tune_model", lambda m, **kw: m)
    monkeypatch.setattr(reg, "tune_model", lambda m, **kw: m)
    monkeypatch.setattr(clf, "compare_models", lambda include, n_select: [Dummy()])
    monkeypatch.setattr(reg, "compare_models", lambda include, n_select: [Dummy()])

def test_hyperparameter_tuning_regression(tmp_path):
    df = pd.DataFrame({"x":[1,2],"y":[3,4]})
    p = tmp_path/"d.csv"; df.to_csv(p,index=False)
    state = {"file_paths":[str(p)], "target_column":"y", "task_type":"regression", "initial_models":["ModelA"]}
    out = HyperparameterTuner.hyperparameter_tuning(state)
    assert "best_model_name" in out and isinstance(out["best_params"], dict)

def test_hyperparameter_tuning_classification(tmp_path):
    df = pd.DataFrame({"x":[1,2],"y":[0,1]})
    p = tmp_path/"d.csv"; df.to_csv(p,index=False)
    state = {"file_paths":[str(p)], "target_column":"y", "task_type":"classification", "initial_models":["ModelA"]}
    out = HyperparameterTuner.hyperparameter_tuning(state)
    assert "best_model_name" in out
