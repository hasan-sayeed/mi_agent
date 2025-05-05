import pytest
from mi_agent.states import GenerateEDAAgentsState, MIExpertState

def test_generate_eda_state_keys():
    s: GenerateEDAAgentsState = {
        "problem_statement": "p",
        "file_paths": [],
        "llm_data_explanations": [],
        "task_type": None,
        "can_merge": None,
        "merge_type": None,
        "target_column": None,
        "feature_columns": None,
        "feature_selection_rationale": [],
        "eda_plans": [],
    }
    assert isinstance(s, dict)

def test_miexpert_state_keys():
    base: dict = {
        "problem_statement": "p",
        "file_paths": [],
        "llm_data_explanations": [],
        "task_type": None,
        "can_merge": None,
        "merge_type": None,
        "target_column": None,
        "feature_columns": None,
        "feature_selection_rationale": [],
        "eda_plans": [],
    }
    s: MIExpertState = {
        **base,
        "explanation": [],
        "initial_models": [],
        "initial_model_results": [],
        "best_model_name": "m",
        "best_params": {},
        "model_explanation": "x",
        "executive_summary": "sum"
    }
    assert "executive_summary" in s
