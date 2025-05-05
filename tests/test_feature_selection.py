import pandas as pd
import pytest
from mi_agent.nodes.feature_selection import FeatureSelector

@pytest.fixture(autouse=True)
def stub_target(monkeypatch):
    import mi_agent.nodes.feature_selection as mod
    class Sel: pass
    sel = Sel()
    sel.target_column = "t"
    sel.feature_columns = ["f1","f2"]
    sel.rationale = "r"
    sel.task_type = "regression"
    monkeypatch.setattr(mod, "target_selector_extractor", type("X", (), {"invoke": staticmethod(lambda prompt: {"responses":[sel]})}))

def test_select_target_and_features(tmp_path):
    df = pd.DataFrame({"f1":[1],"f2":[2],"t":[3]})
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    state = {"file_paths":[str(p)], "problem_statement":"ps"}
    out = FeatureSelector.select_target_and_features(state)
    assert out["target_column"] == "t"
    assert out["feature_columns"] == ["f1","f2"]
    assert out["task_type"] == "regression"
    assert out["feature_selection_rationale"] == ["r"]
