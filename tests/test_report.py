import pytest
from mi_agent.nodes.report import ReportGenerator

@pytest.fixture(autouse=True)
def stub_chat(monkeypatch):
    import mi_agent.nodes.report as mod
    class FakeChat:
        def invoke(self, msgs):
            return type("R",(),{"content":"summary text"})
    monkeypatch.setattr(mod, "ChatOpenAI", lambda model, temperature: FakeChat())

def test_executive_summary_node():
    state = {
        "problem_statement":"ps",
        "llm_data_explanations":["e1"],
        "target_column":"t",
        "feature_columns":["f"],
        "feature_selection_rationale":["r"],
        "eda_plans":[type("P",(),{"name":"p","description":"d"})],
        "explanation":["ex"],
        "initial_models":["m"],
        "best_model_name":"bm",
        "best_params":{"a":1},
        "model_explanation":"me"
    }
    out = ReportGenerator.executive_summary_node(state)
    assert out["executive_summary"] == "summary text"
