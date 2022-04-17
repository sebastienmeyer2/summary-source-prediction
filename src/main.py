"""Main file to run trials and test models."""


from typing import List

from datetime import datetime

import argparse

import numpy as np
import pandas as pd

import optuna


from engine.gridsearch import Objective
from engine.hub import prepare_data
from preprocessing.reader import get_data
from utils.seed_handler import SeedHandler


# pylint: disable=dangerous-default-value
def run(
    seed: int = 42,
    models_names: List[str] = ["logreg"],
    data_path: str = "data/",
    file_suffix: str = "final",
    embeddings_name: str = "glove",
    n_trials: int = 25,
    eval_metric: str = "accuracy",
    save_submission: bool = False
):
    """Run the gridseach and eventually submit predictions.

    Parameters
    ----------
    seed : int, default=42
        The seed to use everywhere for reproducibility.

    models_names : list of str, default=["logreg"]
        Name of the models following project usage. See README.md for more information.

    n_trials : int, default=25
        Number of trials for `optuna` gridsearch.

    eval_metric : str, default="accuracy"
        Which evaluation metric to use. Available metrics are "accuracy" and "f1_weighted".

    save_submission : bool, default=False
        If True, will create a csv file under *results/* folder with predictions to be submitted.
    """
    # Fix seed
    sh = SeedHandler()

    sh.set_seed(seed)
    sh.init_seed()

    # Get data
    train_df, y_train, test_df = get_data(data_path=data_path, file_suffix=file_suffix)

    for model_name in models_names:

        # Transform data if needed
        x_train, x_test, embeddings_matrix = prepare_data(
            model_name, train_df, test_df, embeddings_name=embeddings_name
        )

        # Research grid parameters
        study_id = datetime.now().strftime("%d-%m-%y_%H-%M-%S")

        # Create Objective
        objective = Objective(
            seed, model_name, x_train, y_train,
            embeddings_matrix=embeddings_matrix,
            eval_metric=eval_metric
        )

        # Initialize optuna study object
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(study_name=study_id, direction="maximize", sampler=sampler)

        # Run whole gridsearch
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_trial.params

        # Print summary
        summary: List[str] = []

        summary.append("\n===== OPTUNA GRID SEARCH SUMMARY =====\n")
        summary.append(f"Model is {model_name}.\n")
        summary.append("\n")
        summary.append(f"Cross-validation training {eval_metric}: {objective.best_score}.\n")
        summary.append("\n")
        summary.append(f"Current params are:\n {best_params}\n")
        summary.append("=========================================\n")

        print("".join(summary))

        if save_submission:

            # Create the model with best known parameters and predict
            objective.set_params(best_params)
            y_pred = objective.train_predict(x_test)
            y_sub = pd.DataFrame(y_pred, index=test_df.index)

            # Save file in the requested format
            filename = f"results/submission{study_id}"
            filename += f"_cv_{eval_metric[:2]}_{np.around(objective.best_score, 3)}.csv"
            y_sub.to_csv(filename, header=True)

            print(f"=====> A submission file has been created under {filename}.")
# pylint: enable=dangerous-default-value


if __name__ == "__main__":

    # Command lines
    parser_desc = "Main file to train a model and make predictions."
    parser = argparse.ArgumentParser(description=parser_desc)

    # Seed
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed to use everywhere for reproducbility. Default: 42."
    )

    # Model name
    parser.add_argument(
        "--models-names",
        default=["logreg"],
        nargs="*",
        help="""
             Choose models names. Available models: "rfc", "xgboost", "lightgbm", "catboost",
             "mlp", "logreg", "etc", "stacking" and "embed_lstm".
             """
    )

    # Path to data
    parser.add_argument(
        "--data-path",
        default="data/",
        type=str,
        help="""
             Path to the directory where the data is stored.
             Default: "data/".
             """
    )

    parser.add_argument(
        "--file-suffix",
        default="final",
        type=str,
        help="""
             Suffix to append to the training and test files.
             Default: "final".
             """
    )

    # Number of trials
    parser.add_argument(
        "--trials",
        default=25,
        type=int,
        help="Choose the number of gridsearch trials. Default: 25."
    )

    # Whether to save a submission file
    parser.add_argument(
        "--submission",
        action="store_true",
        help="Use this option to activate submitting a file. Default: Activated."
    )
    parser.add_argument(
        "--no-submission",
        action="store_false",
        dest="submission",
        help="Use this option to deactivate submitting a file. Default: Activated."
    )
    parser.set_defaults(submission=True)

    # Whether to save a submission file
    parser.add_argument(
        "--metric",
        default="accuracy",
        type=str,
        help="""
             Evaluation metric for parameters gridsearch. Available metrics: "accuracy" and
             "f1_weighted". Default: "accuracy".
             """,
        choices=["accuracy", "f1_weighted"]
    )

    # End of command lines
    args = parser.parse_args()

    # Run the gridsearch
    run(
        seed=args.seed,
        models_names=args.models_names,
        data_path=args.data_path,
        file_suffix=args.file_suffix,
        n_trials=args.trials,
        eval_metric=args.metric,
        save_submission=args.submission
    )
