"""Node: run quick PyCaret baseline."""
import pandas as pd
from mi_agent.states import MIExpertState

class QuickBaseline:
    """Run PyCaret’s compare_models(n_select=5) and pull metrics."""

    @staticmethod
    def run_quick_baseline(state: MIExpertState) -> dict:
        """Return `initial_models` list and `initial_model_results` metrics."""
        if state["task_type"] == "classification":
            from pycaret.classification import pull, setup as cls_setup, compare_models as cmp
        else:
            from pycaret.regression import pull, setup as reg_setup, compare_models as cmp
        df = pd.read_csv(state["file_paths"][0])
        if state["task_type"] == "classification":
            cls_setup(df, target=state["target_column"], html=False)
        else:
            reg_setup(df, target=state["target_column"], html=False)
        models = cmp(n_select=5)
        dfm = pull().head(len(models))
        return {
            "initial_models": [type(m).__name__ for m in models],
            "initial_model_results": dfm.to_dict("records"),
        }
