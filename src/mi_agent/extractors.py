"""All TrustCall / LLM‐tool extractor instantiations."""
from langchain_openai import ChatOpenAI
from trustcall import create_extractor
from pydantic import BaseModel, Field
from typing import List, Optional
from mi_agent.app_config import settings

# def get_llm():
#     return ChatOpenAI(
#         model=settings.model_name,
#         temperature=settings.temperature,
#     )

# # shared LLM client (fresh from settings)
# _llm = get_llm()

def get_llm():
    """Always build a fresh ChatOpenAI with the current Settings."""
    return ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
    )


class _LazyExtractor:
    """
    Wrapper around create_extractor that rebinds to a new LLM each time.
    Preserves the .invoke(...) signature.
    """
    def __init__(self, tools, tool_choice, enable_inserts=False):
        self.tools          = tools
        self.tool_choice    = tool_choice
        self.enable_inserts = enable_inserts

    def invoke(self, *args, **kwargs):
        llm       = get_llm()
        extractor = create_extractor(
            llm,
            tools=self.tools,
            tool_choice=self.tool_choice,
            enable_inserts=self.enable_inserts
        )
        return extractor.invoke(*args, **kwargs)



# 2) tool schemas
class FilePath(BaseModel):
    path: str = Field(description="Path to dataset CSV file relevant to the problem.")
class FilePathList(BaseModel):
    file_paths: List[FilePath] = Field(description="List of dataset file paths.")

class EDAPlan(BaseModel):
    name: str = Field(description="A short name for the EDA technique.")
    description: str = Field(description="A one-sentence description of the EDA technique.")
    
    problem_statement: str = Field(
        description="Original research problem statement."
    )
    file_paths: List[str] = Field(
        description="Paths to the (possibly merged) CSVs."
    )
    target_column: str = Field(
        description="Which column we’re predicting."
    )
    feature_columns: List[str] = Field(
        description="Which columns we’ll use as features."
    )
    feature_selection_rationale: str = Field(
        description="The rationale behind how target and feature columns were chosen."
    )

    @property
    def execution_state(self) -> dict:
        return {
            "eda_plan_name":               self.name,
            "eda_plan_description":        self.description,
            "problem_statement":           self.problem_statement,
            "file_paths":                  self.file_paths,
            "target_column":               self.target_column,
            "feature_columns":             self.feature_columns,
            "feature_selection_rationale": self.feature_selection_rationale
        }

class EDAPlanList(BaseModel):
    eda_plans: List[EDAPlan] = Field(description="A comprehensive list of EDA plans with their names and descriptions.")

class MergeDecision(BaseModel):
    can_merge: bool = Field(description="Whether the CSV files can be meaningfully merged.")
    merge_type: Optional[str] = Field(description="Type of merge to apply (e.g., inner, outer, left, right).")
    explanation: str = Field(description="Explain why or why not these files should be merged, and if merging, how and why this strategy is appropriate. Alwayes start with, 'I mm merging the files and saving the data into `merged_data.csv` because...'")

class TargetFeatureSelection(BaseModel):
    target_column: str = Field(description="The column that should be predicted or optimized.")
    feature_columns: List[str] = Field(description="The columns to use as features (inputs) for prediction.")
    rationale: str = Field(description="Explain why these columns were chosen as target and features.")
    task_type: str = Field(description="Type of ML task: 'regression' or 'classification'.")

class EDACodeOutput(BaseModel):
    code: str = Field(description="The Python code to execute for the EDA.")

class EDAExplanation(BaseModel):
    explanation: str = Field(description="A written explanation of the result or error from the EDA code execution.")

# 3) extractors
# file_path_extractor       = create_extractor(_llm, tools=[FilePathList], tool_choice="FilePathList", enable_inserts=True)
# eda_plan_extractor        = create_extractor(_llm, tools=[EDAPlanList],  tool_choice="EDAPlanList",  enable_inserts=True)
# merge_decision_extractor  = create_extractor(_llm, tools=[MergeDecision], tool_choice="MergeDecision", enable_inserts=True)
# target_selector_extractor = create_extractor(_llm, tools=[TargetFeatureSelection], tool_choice="TargetFeatureSelection", enable_inserts=True)
# eda_code_extractor        = create_extractor(_llm, tools=[EDACodeOutput], tool_choice="EDACodeOutput")
# eda_explainer             = create_extractor(_llm, tools=[EDAExplanation],  tool_choice="EDAExplanation")

# 3) lazy extractors
file_path_extractor       = _LazyExtractor([FilePathList],      "FilePathList",      enable_inserts=True)
eda_plan_extractor        = _LazyExtractor([EDAPlanList],       "EDAPlanList",       enable_inserts=True)
merge_decision_extractor  = _LazyExtractor([MergeDecision],     "MergeDecision",     enable_inserts=True)
target_selector_extractor = _LazyExtractor([TargetFeatureSelection], "TargetFeatureSelection", enable_inserts=True)
eda_code_extractor        = _LazyExtractor([EDACodeOutput],     "EDACodeOutput")
eda_explainer             = _LazyExtractor([EDAExplanation],     "EDAExplanation")