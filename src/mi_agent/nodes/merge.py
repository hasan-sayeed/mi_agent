"""Nodes: decide whether to merge and then merge."""
import pandas as pd
from mi_agent.extractors import merge_decision_extractor
from mi_agent.utils import ensure_dir
from mi_agent.app_config import settings
from mi_agent.states import GenerateEDAAgentsState

class MergeNode:
    """Decide & apply a CSV merge."""

    @staticmethod
    def should_merge(state: GenerateEDAAgentsState) -> str:
        """Route-helper: branch into merge_data vs feature_selection."""
        return "merge_data" if state.get("can_merge") else "select_target_and_features"

    @staticmethod
    def can_merge_data(state: GenerateEDAAgentsState) -> dict:
        """Ask LLM if we should merge, and by what key & strategy."""
        if len(state["file_paths"]) <= 1:
            return {"can_merge": False}

        file_summaries = []
        for path in state["file_paths"]:
            df = pd.read_csv(path)
            summary = f"""File: {path}
        Top Rows:
        {df.head(3).to_string(index=False)}

        Columns: {df.columns.tolist()}
        """
            file_summaries.append(summary)

        prompt = f"""
        You are helping a data scientist perform exploratory data analysis (EDA).

        The user has provided the following CSV files:
        {chr(10).join(file_summaries)}

        Below are LLM-generated explanations of the datasets:
        {chr(10).join(state['llm_data_explanations'])}

        Based on the data content and explanations, should these files be merged before EDA?
        If so, suggest the most appropriate merge strategy (e.g., 'inner', 'outer', 'left', 'right').

        Respond using the tool.
        """

        result = merge_decision_extractor.invoke(prompt)
        responses = result.get("responses", [])
        decision = responses[0]

        return {
            "can_merge": decision.can_merge,
            "merge_type": decision.merge_type,
            "llm_data_explanations": [decision.explanation]
        }

    @staticmethod
    def merge_data(state: GenerateEDAAgentsState) -> str:
        """Perform the actual pandas merge into data/merged_data.csv."""
        dfs = [pd.read_csv(p) for p in state["file_paths"]]
        if len(dfs) < 2:
            return {}
        strategy = state.get("merge_type", "inner")
        merged = dfs[0]
        for i in range(1, len(dfs)):
            next_df = dfs[i]
            common_cols = list(set(merged.columns).intersection(set(next_df.columns)))
            if not common_cols:
                raise ValueError(f"No common columns found for merging between file {i} and previous merged result.")
            merged = pd.merge(merged, next_df, how=strategy, on=common_cols)
        ensure_dir(settings.output_dir)
        out = f"{settings.output_dir}/merged_data.csv"
        merged.to_csv(out, index=False)
        return {"file_paths": [out]}
