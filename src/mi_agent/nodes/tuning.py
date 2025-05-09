"""Node: hyperparameter tuning via Optuna in PyCaret."""
import pandas as pd
import numpy as np
from mi_agent.states import MIExpertState
import pycaret.classification as clf
import pycaret.regression as reg


class HyperparameterTuner:
    """Tune each of the baseline models via PyCaret + Optuna."""

    @staticmethod
    def hyperparameter_tuning(state: MIExpertState) -> dict:
        """Return best_model_name, best_model_object, best_params, tuned_model_results."""

        # 1. Load data & state
        df_full  = pd.read_csv(state["file_paths"][0])
        features = state["feature_columns"]
        target   = state["target_column"]
        df       = df_full[features + [target]].copy()
        task     = state["task_type"]
        name_list = state["initial_models"]

        # 2. Build a map Name → ID for PyCaret
        table  = clf.models() if task == "classification" else reg.models()
        id_map = {
            "".join(row["Name"].split()).lower(): model_id
            for model_id, row in table.iterrows()
        }

        # 3. Resolve your names into IDs
        ids = [
            id_map[n.lower().replace(" ", "")]
            for n in name_list
            if n.lower().replace(" ", "") in id_map
        ]

        tuned_results = []
        best_model_obj = None

        if task == "classification":
            # 4a. Set up classification
            clf.setup(df, target=target, html=False)
            tuned = []
            for mid in ids:
                base = clf.create_model(mid)
                try:
                    tuned_model = clf.tune_model(
                        base,
                        n_iter=30,
                        search_library="optuna",
                        search_algorithm="tpe",
                        optimize="Accuracy",
                        choose_better=True,
                        early_stopping=True
                    )
                except ValueError:
                    tuned_model = base
                tuned.append(tuned_model)

            # 5a. Rank *all* tuned models, pull full table, then pick best
            ranked = clf.compare_models(include=tuned, n_select=len(tuned))
            df_tuned = clf.pull()
            tuned_results = df_tuned.to_dict("records")
            best_model_obj = ranked[0] if isinstance(ranked, list) else ranked

        else:
            # 4b. Set up regression
            reg.setup(df, target=target, html=False)
            tuned = []
            for mid in ids:
                base = reg.create_model(mid)
                try:
                    tuned_model = reg.tune_model(
                        base,
                        n_iter=30,
                        search_library="optuna",
                        search_algorithm="tpe",
                        optimize="R2",
                        choose_better=True,
                        early_stopping=True
                    )
                except Exception:
                    tuned_model = base
                tuned.append(tuned_model)

            # 5b. Rank *all* tuned models, pull full table, then pick best
            ranked = reg.compare_models(include=tuned, n_select=len(tuned))
            df_tuned = reg.pull()
            tuned_results = df_tuned.to_dict("records")
            best_model_obj = ranked[0] if isinstance(ranked, list) else ranked

        # 6. Guard against list-wrapped best
        if isinstance(best_model_obj, list):
            best_model_obj = best_model_obj[0]

        # 7. Return everything, including the full tuned-models table
        return {
            "best_model_name":      type(best_model_obj).__name__,
            "best_model_object":    best_model_obj,
            "best_params":          best_model_obj.get_params(),
            "tuned_model_results":  tuned_results
        }
