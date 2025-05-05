"""Node: propose which EDA steps to run."""
from typing import List
from mi_agent.extractors import eda_plan_extractor
from mi_agent.states import GenerateEDAAgentsState
from mi_agent.extractors import EDAPlan

class EDAPlanner:
    """Ask LLM for a list of (name, description) EDA plans."""

    @staticmethod
    def generate_eda_plan(state: GenerateEDAAgentsState) -> dict:
        """Return a list of mi_agent.extractors.EDAPlan models."""
        explanation_text = "\n\n\n\n".join(state["llm_data_explanations"])
        prompt = f"""
        You are an expert data scientist.

        The user has the following dataset explanations:
        {explanation_text}

        In this problem the **target variable** is `{state['target_column']}`, and the following are selected as **features**:
        {', '.join(state['feature_columns'] or [])}

        The rationale behind selecting these columns is: 
        {state['feature_selection_rationale'][0]}

        Now, suggest a list of types of **Exploratory Data Analysis (EDA)** that should be performed to understand the data and the relationship between target and features.

        Each type should have:
        - A short name (e.g., 'correlation analysis')
        - A one-sentence description

        Output should be a list of 3-10 suggestions using the tool.
        """
        result = eda_plan_extractor.invoke(prompt)
        raw = result["responses"][0].eda_plans  # List[EDAPlan] without context
        
        enriched: List[EDAPlan] = []
        for plan in raw:
            enriched.append(
                EDAPlan(
                    name                        = plan.name,
                    description                 = plan.description,
                    problem_statement           = state["problem_statement"],
                    file_paths                  = state["file_paths"],
                    target_column               = state["target_column"] or "",
                    feature_columns             = state["feature_columns"] or [],
                    feature_selection_rationale = state['feature_selection_rationale'][0]
                )
            )
        return {"eda_plans": enriched}
