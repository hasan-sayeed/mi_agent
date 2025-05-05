# tests/test_model_explanation.py

import pytest
from mi_agent.nodes.model_explanation import ModelExplainer

class FakeLLM:
    def invoke(self, messages):
        # messages will be a list of HumanMessage, but we ignore it
        return type("Resp", (), {"content": "explanation text"})()

@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    import mi_agent.nodes.model_explanation as mod
    # replace the _llm object entirely
    monkeypatch.setattr(mod, "_llm", FakeLLM())

def test_ml_explanation_node():
    # create a minimal valid state
    e_plans = [type("P", (), {"name": "p", "description": "d"})]
    state = {
        "problem_statement": "ps",
        "llm_data_explanations": ["exp"],
        "eda_plans": e_plans,
        "explanation": ["ex"],
        "initial_models": ["m1"],
        "initial_model_results": [{"Model": "m1", "MAE": 1, "MSE": 2, "R2": 0.3}],
        "best_model_name": "bm",
        "best_params": {"a": 1}
    }

    out = ModelExplainer.ml_explanation_node(state)
    assert out["model_explanation"] == "explanation text"
