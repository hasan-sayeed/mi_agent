"""Node: build an LLM prompt to explain why we chose the final model."""
from mi_agent.extractors import _llm
from langchain_core.messages import HumanMessage
from mi_agent.states import MIExpertState

class ModelExplainer:
    """Compose the final rationale prompt and call LLM."""

    @staticmethod
    def ml_explanation_node(state: MIExpertState) -> dict:
        """Return `model_explanation` text."""
        prompt = f"""
        You’re an expert data scientist. Here’s the workflow so far:

        Problem:
        {state['problem_statement']}

        Dataset overview:
        {'; '.join(state['llm_data_explanations'])}

        EDA performed:
        """
        # enumerate each EDA plan + its explanation
        for plan, expl in zip(state['eda_plans'], state['explanation']):
            prompt += f"- {plan.name} ({plan.description}): {expl}\n"

        prompt += f"""

        AutoML baseline:
        We ran a quick baseline on these {len(state['initial_models'])} models and got:

        """
        for res in state['initial_model_results']:
            # res is dict with keys like 'Model','MAE','MSE','R2',…
            prompt += "  • {Model}: MAE={MAE:.3f}, MSE={MSE:.3f}, R2={R2:.3f}\n".format(**res)

        prompt += f"""

        Hyperparameter tuning:
        Final chosen model: {state['best_model_name']}
        Tuned parameters: {state['best_params']}

        Please write a clear and brief rationale that covers:
        1. Why AutoML selected those initial models.
        2. How the EDA insights supported that choice.
        3. Why the final model outperformed the others.
        4. What the tuned hyperparameters reveal about the data/model.
        """

        resp = _llm.invoke([HumanMessage(content=prompt)])
        return {"model_explanation": resp.content}
