# tests/test_data_explainer.py

import pandas as pd
import pytest
from mi_agent.nodes.data_explainer import DataExplainer

class DummyResp:
    def __init__(self, content):
        self.content = content

class FakeLLM:
    def invoke(self, prompt):
        return DummyResp("explained")

@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    # override the module‐level _llm with our fake
    import mi_agent.nodes.data_explainer as mod
    monkeypatch.setattr(mod, "_llm", FakeLLM())

def test_read_and_explain_data(tmp_path):
    # create two small csvs
    p1 = tmp_path / "one.csv"
    p2 = tmp_path / "two.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(p1, index=False)
    pd.DataFrame({"y": [3, 4]}).to_csv(p2, index=False)

    state = {"file_paths": [str(p1), str(p2)], "problem_statement": "ps"}
    out = DataExplainer.read_and_explain_data(state)

    # both paths should be "explained"
    assert out["llm_data_explanations"] == ["explained", "explained"]
