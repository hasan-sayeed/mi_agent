import pytest
from mi_agent.nodes.eda_plan import EDAPlanner

class DummyPlan:
    def __init__(self, name, description):
        self.name = name
        self.description = description

@pytest.fixture(autouse=True)
def stub_plan(monkeypatch):
    import mi_agent.nodes.eda_plan as mod
    def fake_invoke(prompt):
        return {"responses": [type("R", (), {"eda_plans": [DummyPlan("p1","d1"), DummyPlan("p2","d2")]})]}
    monkeypatch.setattr(mod, "eda_plan_extractor", type("X", (), {"invoke": staticmethod(fake_invoke)}))

def test_generate_eda_plan():
    state = {
        "llm_data_explanations": ["e1","e2"],
        "target_column": "t",
        "feature_columns": ["f"],
        "feature_selection_rationale": ["r"],
        "problem_statement": "ps",
        "file_paths": ["a.csv"]
    }
    out = EDAPlanner.generate_eda_plan(state)
    plans = out["eda_plans"]
    assert plans[0].name == "p1"
    assert plans[1].description == "d2"
