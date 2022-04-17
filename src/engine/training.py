"""Training and evaluation functions for general models."""


from typing import Any, Dict, Optional, Union

from numpy import ndarray
from pandas import DataFrame

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


from engine.hub import create_model


def train_predict(
    model_name: str, params: Dict[str, Any], x_train: Union[ndarray, DataFrame],
    x_eval: Union[ndarray, DataFrame], y_train: Union[ndarray, DataFrame],
    embeddings_matrix: Optional[ndarray] = None
) -> ndarray:
    """Fit a model on training set and predict on eval set.

    The model is expected to possess the fit and predict methods.

    Parameters
    ----------
    model_name : str
        The name of model following project usage. See README.md for more information about
        available models.

    params : dict of {str: any}
        A dictionary of parameters for chosen **model_name**.

    x_train : ndarray or DataFrame
        Training features.

    x_eval : ndarray or DataFrame
        Evaluation features.

    y_train : ndarray or DataFrame
        Training labels.

    embeddings_matrix : optional ndarray, default=None
        Matrix of tokens to embeddings for embed models.

    Returns
    -------
    y_pred : ndarray
        Predicted labels.
    """
    # Initialize model
    model_raw = create_model(model_name, params, embeddings_matrix=embeddings_matrix)

    # Fit on whole training set
    if isinstance(y_train, DataFrame):
        y_train = y_train.to_numpy().flatten()
    model_raw.fit(x_train, y_train)

    # Predict new values
    y_pred = model_raw.predict(x_eval)

    return y_pred


def train_eval(
    model_name: str, params: Dict[str, Any], x_train: Union[ndarray, DataFrame],
    x_eval: Union[ndarray, DataFrame], y_train: Union[ndarray, DataFrame],
    y_eval: Union[ndarray, DataFrame], embeddings_matrix: Optional[ndarray] = None,
    eval_metric: str = "accuracy"
) -> float:
    """Fit a model on training set and compute metrics on evaluation set.

    The model is expected to possess the fit and predict methods.

    Parameters
    ----------
    model_name : str
        The name of model following project usage. See README.md for more information about
        available models.

    params : dict of {str: any}
        A dictionary of parameters for chosen **model_name**.

    x_train : ndarray or DataFrame
        Training features.

    x_eval : ndarray or DataFrame
        Evaluation features.

    y_train : ndarray or DataFrame
        Training labels.

    y_eval : ndarray or DataFrame
        Evaluation labels.

    embeddings_matrix : optional ndarray, default=None
        Matrix of tokens to embeddings for embed models.

    eval_metric : str, default="accuracy"
        Which evaluation metric to use. Available metrics are "accuracy" and "f1_weighted".

    Returns
    -------
    score : float
        **eval_metric** of chosen **model_name** and **params** on the fold.

    Raises
    ------
    ValueError
        If the **eval_metric** is unsupported.
    """
    # Predict new values
    y_pred = train_predict(
        model_name, params, x_train, x_eval, y_train,
        embeddings_matrix=embeddings_matrix
    )

    # Evaluate the model
    if eval_metric == "accuracy":
        score = accuracy_score(y_eval, y_pred)
    elif eval_metric == "f1_weighted":
        score = f1_score(y_eval, y_pred, average="weighted")
    else:
        err_msg = f"Unsupported eval metric {eval_metric}."
        err_msg += " Please choose from accuracy and f1_weighted."
        raise ValueError(err_msg)

    return score


def cross_val(
    model_name: str, params: Dict[str, Any], x_train: Union[ndarray, DataFrame],
    y_train: Union[ndarray, DataFrame], embeddings_matrix: Optional[ndarray] = None,
    eval_metric: str = "accuracy", n_folds: int = 5
) -> float:
    """Run cross-validation using `StratifiedKFold` splitter.

    Parameters
    ----------
    model_name : str
        The name of model following project usage. See README.md for more information about
        available models.

    params : dict of {str: any}
        A dictionary of parameters for chosen **model_name**.

    x_train : ndarray or DataFrame
        Training features.

    y_train : ndarray or DataFrame
        Training labels.

    embeddings_matrix : optional ndarray, default=None
        Matrix of tokens to embeddings for embed models.

    eval_metric : str, default="accuracy"
        Which evaluation metric to use. Available metrics are "accuracy" and "f1_weighted".

    n_folds : int, default=5
        Number of splits for cross-validation.

    Returns
    -------
    mean_score : float
        Mean score on the cross validation.
    """
    # Compute mean metrics
    mean_score = 0.

    # Cross validation parameters
    kf = StratifiedKFold(n_splits=n_folds)

    for train_indices, eval_indices in kf.split(x_train, y_train):

        # Data splitting
        if isinstance(x_train, DataFrame):
            fold_x_train, fold_x_eval = x_train.iloc[train_indices], x_train.iloc[eval_indices]
        else:
            fold_x_train, fold_x_eval = x_train[train_indices], x_train[eval_indices]

        if isinstance(y_train, DataFrame):
            fold_y_train, fold_y_eval = y_train.iloc[train_indices], y_train.iloc[eval_indices]
        else:
            fold_y_train, fold_y_eval = y_train[train_indices], y_train[eval_indices]

        # Fold evaluation
        fold_eval_score = train_eval(
            model_name, params, fold_x_train, fold_x_eval, fold_y_train, fold_y_eval,
            embeddings_matrix=embeddings_matrix, eval_metric=eval_metric
        )

        mean_score += fold_eval_score

    # Compute mean values of selected metrics
    mean_score /= n_folds

    return mean_score
