"""Node: pick target & features."""
import pandas as pd
from mi_agent.extractors import target_selector_extractor
from mi_agent.states import GenerateEDAAgentsState

class FeatureSelector:
    """Ask LLM to choose a target and features."""

    @staticmethod
    def select_target_and_features(state: GenerateEDAAgentsState) -> dict:
        """Return target_column, feature_columns, rationale, and task_type."""
        file_summaries = []
        for path in state["file_paths"]:
            df = pd.read_csv(path)
            summary = f"""File: {path}
        Top Rows:
        {df.head(5).to_string(index=False)}

        Columns: {df.columns.tolist()}
        """
            file_summaries.append(summary)

        prompt = f"""
        You are helping select the target and feature columns for a data science task.

        Problem statement:
        \"\"\"{state['problem_statement']}\"\"\"

        Dataset previews:
        {chr(10).join(file_summaries)}

        Based on the above, choose:
        - A prediction target column
        - The feature columns to use as input
        - Explain your reasoning behind choosing these columns
        - And finally, decide whether the appropriate ML task is 'regression' or 'classification'.

        Use the tool to respond.
        """

        result = target_selector_extractor.invoke(prompt)
        responses = result.get("responses", [])
        if not responses:
            return {
                "target_column": None,
                "feature_columns": [],
                "feature_selection_rationale": ["LLM failed to respond."]
            }

        selection = responses[0]
        return {
            "target_column": selection.target_column,
            "feature_columns": selection.feature_columns,
            "feature_selection_rationale": [selection.rationale],
            "task_type": selection.task_type,
        }
