"""Node: hyperparameter tuning via Optuna in PyCaret."""
import pandas as pd
import numpy as np
from mi_agent.states import MIExpertState
import pycaret.classification as clf
import pycaret.regression as reg


class HyperparameterTuner:
    """Tune each of the 5 baseline models via PyCaret + Optuna."""

    @staticmethod
    def hyperparameter_tuning(state: MIExpertState) -> dict:
        """Return best_model_name, best_model_object, best_params."""

        # load full DataFrame and then sub-select features + target
        df_full   = pd.read_csv(state["file_paths"][0])
        features  = state["feature_columns"]       # list of column names
        target    = state["target_column"]
        # build a clean df with only selected features + target
        df        = df_full[features + [target]].copy()
        task      = state["task_type"]
        name_list = state["initial_models"]

        # build lookup Name → ID
        table  = clf.models() if task == "classification" else reg.models()
        id_map = {
            "".join(row["Name"].split()).lower(): model_id
            for model_id, row in table.iterrows()
        }

        # resolve names → IDs
        ids = [
            id_map[n.lower().replace(" ", "")]
            for n in name_list
            if n.lower().replace(" ", "") in id_map
        ]

        # re-initialize PyCaret & tune
        if task == "classification":
            clf.setup(df, target=target, html=False)
            tuned = []
            for mid in ids:
                base = clf.create_model(mid)
                try:
                    tuned_model = clf.tune_model(
                        base,
                        n_iter=30,
                        search_library='optuna',
                        search_algorithm='tpe',
                        optimize='Accuracy',
                        choose_better=True,
                        fit_kwargs={"error_score": np.nan}
                    )
                except ValueError:
                    # if tuning itself fails, fall back to the base model
                    tuned_model = base
                tuned.append(tuned_model)
            best = clf.compare_models(include=tuned, n_select=1)

        else:
            reg.setup(df, target=target, html=False)
            tuned = []
            for mid in ids:
                base = reg.create_model(mid)
                try:
                    tuned_model = reg.tune_model(
                        base,
                        n_iter=5,
                        search_library='optuna',
                        search_algorithm='tpe',
                        optimize='R2',
                        choose_better=True
                    )
                    print(f"Tuning passed!!!!!!!!!")
                except Exception as e:
                    print(f"Tuning failed ({e}), falling back to baseline.")
                    tuned_model = base
                tuned.append(tuned_model)
            best = reg.compare_models(include=tuned, n_select=1)

        if isinstance(best, list):
            best = best[0]

        return {
            "best_model_name": type(best).__name__,
            "best_model_object": best,
            "best_params": best.get_params()
        }
