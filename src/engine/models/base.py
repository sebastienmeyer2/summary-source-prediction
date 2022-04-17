"""Wrapper for base models."""


from typing import Any, Dict, List, Union

from numpy import ndarray
from pandas import DataFrame

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from mlxtend.classifier import StackingCVClassifier


class BaseEstimator():
    """A wrapper for machine learning models."""

    def __init__(self, model_name: str, params: Dict[str, Any]):
        """The constructor of the class.

        Parameters
        ----------
        model_name : str
            The name of model following project usage. See README.md for more information.

        params : dict of {str: any}
            A dictionary of parameters for chosen **model_name**. It contains all parameters to
            initialize and fit the model.
        """
        self.model_name = model_name

        # Model parameters
        self.model_params = params

        # Label prediction
        if model_name == "rfc":
            self.model = RandomForestClassifier(**self.model_params)
        elif model_name == "etc":
            self.model = ExtraTreesClassifier(**self.model_params)
        elif model_name == "xgboost":
            self.model = XGBClassifier(**self.model_params)
        elif model_name == "lightgbm":
            self.model = LGBMClassifier(**self.model_params)
        elif model_name == "catboost":
            self.model = CatBoostClassifier(**self.model_params)
        elif model_name == "logreg":
            self.model = LogisticRegression(**self.model_params)
        elif self.model_name == "stacking":
            final_est = LogisticRegression(random_state=self.model_params["seed"])
            self.model = StackingCVClassifier(
                create_est(self.model_params), final_est, use_probas=False, cv=5,
                random_state=self.model_params["seed"], n_jobs=1, verbose=1
            )
        else:
            raise ValueError(f"{model_name} is not implemented. Check available models.")

    def fit(self, x_train: Union[ndarray, DataFrame], y_train: Union[ndarray, DataFrame]):
        """Wrapper of fit method.

        Parameters
        ----------
        x_train : ndarray or DataFrame
            Training features.

        y_train : ndarray or DataFrame
            Training labels.
        """
        self.model.fit(x_train, y_train)

    def predict(self, x_eval: Union[ndarray, DataFrame]) -> Union[ndarray, DataFrame]:
        """Wrapper of predict method.

        Parameters
        ----------
        x_eval : ndarray or DataFrame
            Evaluation features.

        Returns
        -------
        y_pred : ndarray or DataFrame
            Predicted labels on evaluation set.
        """
        y_pred = self.model.predict(x_eval)

        return y_pred

    def predict_proba(self, x_eval: Union[ndarray, DataFrame]) -> Union[ndarray, DataFrame]:
        """Wrapper of predict_proba method.

        Parameters
        ----------
        x_eval : ndarray or DataFrame
            Evaluation features.

        Returns
        -------
        y_pred_probs : ndarray or DataFrame
            Predicted probabilities on evaluation set.
        """
        y_pred_probs = self.model.predict_proba(x_eval)

        return y_pred_probs


def create_est(params: Dict[str, Any]) -> List[Any]:
    """Build up estimators for stacking.

    Parameters
    ----------
    params : dict of {str: any}
        A dictionary of parameters for stacking. It contains all parameters to initialize and fit
        the model.

    Returns
    -------
    est : list of any
        The complete sequence of estimators for stacking.
    """
    seed = params["random_state"]

    # For now, manually estimated parameters
    rfc = RandomForestClassifier(
        max_samples=None, bootstrap=True, random_state=seed, n_jobs=-1, n_estimators=234,
        criterion="entropy", max_depth=None, min_samples_split=3, min_samples_leaf=3,
        max_features=0.4, ccp_alpha=1.11e-6
    )
    lgbm = LGBMClassifier(
        objective="binary", verbose=-1, random_state=seed, n_jobs=-1, n_estimators=208,
        num_leaves=94, min_split_gain=2.65e-4, min_child_weight=4.70e-5, min_child_samples=10,
        subsample=0.935, subsample_freq=3, reg_alpha=8.94e-6, reg_lambda=7.77e-6
    )
    xgb = XGBClassifier(
        eval_metric="logloss", use_label_encoder=False, random_state=seed, n_estimators=241,
        subsample=0.828, colsample_bytree=0.84, colsample_bylevel=0.73, colsample_bynode=0.75,
        max_delta_step=0.30, reg_alpha=1.10e-7, reg_lambda=1.05e-7, grow_policy="depthwise",
        tree_method="gpu_hist", gpu_id=0, sampling_method="gradient_based"
    )
    cat = CatBoostClassifier(
        verbose=100, allow_writing_files=False, random_state=seed, n_estimators=2442,
        learning_rate=0.022, grow_policy="Depthwise", bootstrap_type="Bernoulli",
        leaf_estimation_iterations=10, leaf_estimation_backtracking="AnyImprovement",
        od_type="Iter", l2_leaf_reg=9.21e-4, min_data_in_leaf=2, subsample=0.62
    )

    est = [rfc, lgbm, xgb, cat]

    return est
