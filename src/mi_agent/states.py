"""Shared pipeline state definitions for MI-Agent."""
from typing import TypedDict, List, Optional, Dict, Any
from typing_extensions import Annotated
import operator

from mi_agent.extractors import EDAPlan

class GenerateEDAAgentsState(TypedDict):
    problem_statement: str
    file_paths: List[str]
    llm_data_explanations: Annotated[List[str], operator.add]
    task_type: Optional[str]
    can_merge: Optional[bool]
    merge_type: Optional[str]
    target_column: Optional[str]
    feature_columns: Optional[List[str]]
    feature_selection_rationale: Annotated[List[str], operator.add]
    eda_plans: List[EDAPlan]

class EDAExecutionState(TypedDict):
    plan: EDAPlan
    max_retries: int
    tries: int
    generated_code: Optional[str]
    code_output: Optional[str]
    code_file_path: Optional[str]
    image_path: Optional[str]
    explanation: str

class MIExpertState(TypedDict):
    problem_statement: str
    file_paths: List[str]
    llm_data_explanations: Annotated[List[str], operator.add]
    task_type: Optional[str]
    can_merge: Optional[bool]
    merge_type: Optional[str]
    target_column: Optional[str]
    feature_columns: Optional[List[str]]
    feature_selection_rationale: Annotated[List[str], operator.add]
    eda_plans: List[EDAPlan]
    explanation: Annotated[List[str], operator.add]
    initial_models: List[str]
    initial_model_results: List[Dict[str, Any]]
    best_model_name: str
    best_params: Dict[str, Any]
    model_explanation: str
    technical_summary: str