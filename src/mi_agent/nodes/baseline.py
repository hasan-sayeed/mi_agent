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
        # load full DataFrame and then sub-select features + target
        df_full   = pd.read_csv(state["file_paths"][0])
        features  = state["feature_columns"]       # list of column names
        target    = state["target_column"]
        # build a clean df with only selected features + target
        df        = df_full[features + [target]].copy()
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
