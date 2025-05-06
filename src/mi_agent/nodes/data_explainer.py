"""Node: have the LLM explain each CSV’s contents."""
import pandas as pd
from typing import List
from langchain_openai import ChatOpenAI
from mi_agent.extractors import get_llm
from mi_agent.states import GenerateEDAAgentsState

class DataExplainer:
    """Load each CSV, show head(3), let LLM explain."""

    @staticmethod
    def read_and_explain_data(state: GenerateEDAAgentsState) -> dict:
        """Return LLM descriptions for each path."""
        explanations: List[str] = []
        _llm = get_llm()
        for path in state["file_paths"]:
            df = pd.read_csv(path)
            prompt = f"""
            The following data is loaded from {path}:
            {df.head(3).to_string()}

            Columns: {df.columns.tolist()}

            Based on this and the problem: \"{state['problem_statement']}\",
            explain what kind of data this is, and what it might be used for.
            Mention the name of the file which you are talking about.
            """
            explanation = _llm.invoke(prompt)
            explanations.append(explanation.content)
        return {"llm_data_explanations": explanations}
