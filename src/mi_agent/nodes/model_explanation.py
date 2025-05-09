"""Node: build an LLM prompt to explain why we chose the final model."""
from mi_agent.extractors import get_llm
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
        We then tuned hyperparameters of each of those models and obtained:
        """
        for res in state['tuned_model_results']:
            prompt += "  • {Model}: MAE={MAE:.3f}, MSE={MSE:.3f}, R2={R2:.3f}\n".format(**res)

        # --- Final choice ---
        prompt += f"""
        Final chosen model: {state['best_model_name']}
        Tuned parameters: {state['best_params']}

        Please write a clear, structured rationale that covers:
        1. **Why** AutoML’s initial ranking picked those baselines (point to their MAE/MSE/R2 etc. elaborately).  
        2. **How** our EDA insights reinforce or explain that ranking.  
        3. **How** the hyperparameter tuning shifted performance (reference specific metric changes elaborately in the tuned table).  
        4. **Why** the final model came out on top (tie back to both its baseline and tuned metrics).  
        5. **What** the specific tuned parameters reveal about the data or modeling choices.  
        6. **What to expect** from the final model in deployment—its strengths, assumptions, and any limitations.
        """

        _llm = get_llm()
        resp = _llm.invoke([HumanMessage(content=prompt)])
        return {"model_explanation": resp.content}
