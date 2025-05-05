"""Node: generate a two-page executive summary."""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from mi_agent.states import MIExpertState

class ReportGenerator:
    """Write the two-page `executive_summary`."""

    @staticmethod
    def executive_summary_node(state: MIExpertState) -> dict:
        """Return `executive_summary` Markdown."""
        prompt = f"""
        You’re a data science engineer writing a **two-page executive summary** for senior management. Clients come to your company to solves their data science problems.
        Be concise but complete:

        **1. Problem Statement**  
        {state['problem_statement']}

        **2. Data Provided**  
        We have the following datasets and their contents:
        """
        for expl in state['llm_data_explanations']:
            prompt += f"- {expl}\n"

        prompt += f"""

        **3. Target & Feature Selection**  
        - **Target column:** {state['target_column']}  
        - **Feature columns:** {', '.join(state['feature_columns'] or [])}  
        - **Rationale:** {state['feature_selection_rationale'][0]}
        Briefly talk about what columns were chosen as features and which was chosen as target. Why?
        
        **4. EDA Results**  
        For each EDA plan, give its name, one-sentence description, and 2-3 sentence takeaway:
        """
        for plan, expl in zip(state['eda_plans'], state['explanation']):
            prompt += f"- **{plan.name}** ({plan.description}): {expl}\n"

        prompt += f"""

        **5. Model Selection & Performance**  
        - We used AutoML to evaluate a broad suite of models and identified these {len(state['initial_models'])} as the top-performing baselines:  
        {', '.join(state['initial_models'])}.  
        - We then performed hyperparameter tuning on each of those models.  
        - The final chosen model is **{state['best_model_name']}**, with tuned parameters: {state['best_params']}.  
        - **Why it excelled**: tie this back to your EDA insights (e.g. features with strong correlations, robustness to outliers, data distribution characteristics).
        Here is a broad exploration of model selection and performance that was written before: {state['model_explanation']}. You can follow this if you want.

        **6. Recommendations**  
        Conclude with **1-5 bullet-point** recommendations on how to improve future model performance (ex. data to collect, features to engineer, or process changes) based on the report so far.

        Write this in clear, non-technical language suitable for a two-page handout.

        At last, include any 'Prepared By: The Agnetic System designed by Hasan Sayeed.'.
        """

        llm_executive_summary_node = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
        resp = llm_executive_summary_node.invoke([HumanMessage(content=prompt)])
        return {"executive_summary": resp.content}