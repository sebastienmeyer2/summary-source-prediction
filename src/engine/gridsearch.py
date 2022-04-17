"""Run optimized cross validation for parameters gridsearch."""


from typing import Any, Dict, Optional, Union

from numpy import ndarray
from pandas import DataFrame

from optuna.trial import Trial


from engine.training import cross_val, train_predict


class Objective():
    """An Objective class to wrap trials.

    General class that implements call functions for gridsearch algorithms.
    """
    def __init__(
        self, seed: int, model_name: str, x_train: Union[ndarray, DataFrame],
        y_train: Union[ndarray, DataFrame], embeddings_matrix: Optional[ndarray] = None,
        eval_metric: str = "accuracy"
    ):
        """The constructor of the class.

        Parameters
        ----------
        seed : int
            The seed to use everywhere for reproducibility.

        model_name : str
            The name of model following project usage. See README.md for more information.

        x_train : ndarray or DataFrame
            Training features.

        y_train : ndarray or DataFrame
            Training labels.

        embeddings_matrix : optional ndarray, default=None
            Matrix of tokens to embeddings for embed models.

        eval_metric : str, default="accuracy"
            Which evaluation metric to use. Available metrics are "accuracy" and "f1_weighted".
        """
        # Handling randomness
        self.seed = seed

        # Data
        self.x_train = x_train
        self.y_train = y_train
        self.input_dim = self.x_train.shape[1]
        self.embeddings_matrix = embeddings_matrix

        # Model and decision values
        self.model_name = model_name

        # Model parameters
        self.params: Dict[str, Any] = {}

        # Keep best results in memory
        self.eval_metric = eval_metric
        self.best_score = 0.

    def get_params(self) -> Dict[str, Any]:
        """Get parameters for associated model.

        Returns
        -------
        params : dict of {str: any}
            A dictionary of parameters for chosen **model_name**. Initially, this attribute is
            empty and it is changed as many times as there are trials.
        """
        return self.params

    def set_params(self, params: Dict[str, Any]):
        """Set parameters for associated model.

        Parameters
        ----------
        params : dict of {str: any}
            A dictionary of parameters for chosen **model_name**. It contains all parameters to
            initialize and fit the model.
        """
        self.params = params

    def run_cross_val(self) -> float:
        """Initialize a model and run cross-validation on training set.

        Returns
        -------
        mean_score : float
            Mean target metric value of current model during cross-validation.
        """
        # Run cross-validation
        mean_score = cross_val(
            self.model_name, self.params, self.x_train, self.y_train,
            embeddings_matrix=self.embeddings_matrix, eval_metric=self.eval_metric
        )

        # Keep best values in memory
        if mean_score > self.best_score:

            self.best_score = mean_score

        return mean_score

    def train_predict(self, x_eval: Union[DataFrame, ndarray]) -> ndarray:
        """Initialize a model and predict on test set.

        Parameters
        ----------
        x_eval : ndarray or DataFrame
            Evaluation features.

        Returns
        -------
        y_pred : ndarray
            Predicted labels.
        """
        y_pred = train_predict(
            self.model_name, self.params, self.x_train, x_eval, self.y_train,
            embeddings_matrix=self.embeddings_matrix
        )

        return y_pred

    def __call__(self, trial: Trial) -> float:
        """Run a trial using `optuna` package.

        Parameters
        ----------
        trial : Trial
            An instance of `Trial` object from `optuna` package to handle parameters search.

        Returns
        ----------
        trial_target : float
            Target metric value of current model during trial.
        """
        # Initialize parameter grid via optuna
        optuna_params = optuna_param_grid(trial, self.seed, self.model_name, self.input_dim)

        self.set_params(optuna_params)

        # Run cross val evaluation
        trial_target = self.run_cross_val()

        return trial_target


def optuna_param_grid(
    trial: Trial, seed: int, model_name: str, input_dim: int, use_gpu: bool = True
) -> Dict[str, Any]:
    """Create a param grid for `optuna` usage.

    Parameters
    ----------
    trial : Trial
        An instance of `Trial` object from `optuna` package to handle parameters search.

    seed : int
        The seed to use everywhere for reproducibility.

    model_name : str
        The name of model following project usage. See README.md for more information.

    input_dim : int
        Number of features.

    use_gpu : bool, default=True
        If True, will select parameters to run on GPU.

    Returns
    -------
    params : dict of {str: any}
        A dictionary of parameters for chosen **model_name**. It contains all parameters to
        initialize and fit the model.

    Raises
    ------
    ValueError
        If no gridsearch corresponds to the **model_name**.
    """
    # Initialize model params
    params = {}

    if model_name == "logreg":

        # Logistic Regression

        # Parameters ensuring reproducibility and overall adaptability
        params["random_state"] = trial.suggest_categorical("random_state", [seed])
        params["max_iter"] = trial.suggest_int("max_iter", 100, 500)

        # Main parameters
        params["solver"] = trial.suggest_categorical("solver", ["newton-cg"])
        params["penalty"] = trial.suggest_categorical("penalty", ["l2"])

        if params["penalty"] == "l2":
            params["C"] = trial.suggest_float("C", 1e-1, 1e4, log=True)

        # End of Logistic Regression

    elif model_name == "rfc":  # same types of parameters

        # Random Forest

        # Parameters ensuring reproducibility and overall adaptability
        params["max_samples"] = trial.suggest_categorical("max_samples", [None])
        params["bootstrap"] = trial.suggest_categorical("bootstrap", [True])
        params["random_state"] = trial.suggest_categorical("random_state", [seed])
        params["n_jobs"] = trial.suggest_categorical("n_jobs", [-1])

        # Main parameters
        params["n_estimators"] = trial.suggest_int("n_estimators", 30, 500)
        params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
        params["max_depth"] = trial.suggest_categorical("max_depth", [5, 6, 7, None])
        params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 8)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 10)
        params["max_features"] = trial.suggest_categorical("max_features", ["auto", 0.4, 0.8])
        params["ccp_alpha"] = trial.suggest_float("ccp_alpha", 1e-8, 1e-5, log=True)

        # End of Random Forest

    elif model_name in "etc":

        # Extra Trees

        # Parameters ensuring reproducibility and overall adaptability
        params["random_state"] = trial.suggest_categorical("random_state", [seed])
        params["n_jobs"] = trial.suggest_categorical("n_jobs", [-1])

        # Main parameters
        params["n_estimators"] = trial.suggest_int("n_estimators", 200, 500)
        params["criterion"] = trial.suggest_categorical("criterion", ["gini"])
        params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 4)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 2)
        params["min_impurity_decrease"] = \
            trial.suggest_float("min_impurity_decrease", 1e-9, 1e-6, log=True)
        params["ccp_alpha"] = trial.suggest_float("ccp_alpha", 1e-8, 1e-5, log=True)

        # End of Extra Trees

    elif model_name == "xgboost":

        # XGBoost

        # Parameters ensuring reproducibility and overall adaptability
        params["eval_metric"] = trial.suggest_categorical("eval_metric", ["logloss"])
        params["use_label_encoder"] = trial.suggest_categorical("use_label_encoder", [False])
        params["random_state"] = trial.suggest_categorical("random_state", [seed])

        # Main parameters
        params["n_estimators"] = trial.suggest_int("n_estimators", 30, 250)
        # params["max_depth"] = trial.suggest_int("max_depth", 2, 6)
        params["gamma"] = trial.suggest_float("gamma", 1e-3, 1e-1, log=False)
        params["subsample"] = trial.suggest_float("subsample", 0.2, 1.0, log=False)
        params["colsample_bytree"] = \
            trial.suggest_float("colsample_bytree", 0.6, 1.0, log=False)
        params["colsample_bylevel"] = \
            trial.suggest_float("colsample_bylevel", 0.6, 1.0, log=False)
        params["colsample_bynode"] = \
            trial.suggest_float("colsample_bynode", 0.6, 1.0, log=False)
        params["max_delta_step"] = trial.suggest_float("max_delta_step", 0.0, 1, log=False)
        params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-7, 1e-4, log=True)
        params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-7, 1e-4, log=True)
        params["grow_policy"] = \
            trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        if use_gpu:

            params["tree_method"] = trial.suggest_categorical("tree_method", ["gpu_hist"])
            params["gpu_id"] = trial.suggest_categorical("gpu_id", [0])
            params["sampling_method"] = \
                trial.suggest_categorical("sampling_method", ["gradient_based"])

        # End of XGBoost

    elif model_name == "catboost":

        # CatBoost

        # Parameters ensuring reproducibility and overall adaptability
        params["verbose"] = trial.suggest_categorical("verbose", [100])
        params["allow_writing_files"] = \
            trial.suggest_categorical("allow_writing_files", [False])
        params["random_state"] = trial.suggest_categorical("random_state", [seed])

        # Main parameters
        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 2500)
        params["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
        params["grow_policy"] = \
            trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"])
        # params["depth"] = trial.suggest_int("depth", 4, 8)
        params["bootstrap_type"] = \
            trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
        params["leaf_estimation_iterations"] = \
            trial.suggest_int("leaf_estimation_iterations", 3, 15)
        params["leaf_estimation_backtracking"] = \
            trial.suggest_categorical("leaf_estimation_backtracking", ["No", "AnyImprovement"])
        params["od_type"] = trial.suggest_categorical("od_type", ["IncToDec", "Iter"])
        params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1e-7, 1e-2, log=True)

        # Grow policy parameters
        if params["grow_policy"] in {"Lossguide", "Depthwise"}:
            params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 10)
        if params["grow_policy"] == "SymmetricTree":
            params["boosting_type"] = \
                trial.suggest_categorical("boosting_type", ["Ordered", "Plain"])
            if params["boosting_type"] == "Plain":
                params["score_function"] = \
                    trial.suggest_categorical("score_function", ["Cosine", "L2"])

        # Bootstrap parameters
        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = \
                trial.suggest_float("bagging_temperature", 1e-2, 5, log=True)
        else:
            params["subsample"] = trial.suggest_float("subsample", 0.4, 1, log=False)

        # End of CatBoost

    elif model_name == "lightgbm":

        # LightGBM

        # Parameters ensuring reproducibility and overall adaptability
        params["objective"] = trial.suggest_categorical("objective", ["binary"])
        params["verbose"] = trial.suggest_categorical("verbose", [-1])
        params["random_state"] = trial.suggest_categorical("seed", [seed])
        params["n_jobs"] = trial.suggest_categorical("n_jobs", [-1])

        # Main parameters
        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 250)
        params["num_leaves"] = trial.suggest_int("num_leaves", 10, 100)
        params["min_split_gain"] = trial.suggest_float("min_split_gain", 1e-6, 1e-2, log=True)
        params["min_child_weight"] = \
            trial.suggest_float("min_child_weight", 1e-7, 1e-4, log=True)
        params["min_child_samples"] = trial.suggest_int("min_child_samples", 2, 15)
        params["subsample"] = trial.suggest_float("subsample", 0.2, 1.0, log=False)
        params["subsample_freq"] = trial.suggest_int("subsample_freq", 0, 6)
        params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-6, 1e-3, log=True)
        params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-6, 1e-3, log=True)

        # End of LightGBM

    elif model_name == "mlp":

        # Multi-Layer Perceptron

        # Main parameters
        params["random_state"] = trial.suggest_categorical("random_state", [seed])
        params["nb_hidden_layers"] = trial.suggest_int("nb_hidden_layers", 1, 4)
        params["hidden_dim"] = trial.suggest_int("hidden_dim", 50, 400)
        params["hidden_act"] = \
            trial.suggest_categorical("hidden_act", ["sigmoid", "tanh", "relu"])
        params["hidden_init"] = \
            trial.suggest_categorical("hidden_init", ["uniform", "glorot_uniform"])
        params["input_dim"] = trial.suggest_categorical("input_dim", [input_dim])

        # Optimizer parameters
        params["learning_rate"] = trial.suggest_float("learning_rate", 5e-3, 1e-1, log=True)
        params["optim_name"] = trial.suggest_categorical("optim_name", ["adam", "rmsprop"])
        if params["optim_name"] == "adam":
            params["beta_1"] = trial.suggest_categorical("beta_1", [0.9])
            params["beta_2"] = trial.suggest_categorical("beta_2", [0.999])
        elif params["optim_name"] == "rmsprop":
            params["rho"] = trial.suggest_float("rho", 0.8, 0.95)
            params["momentum"] = trial.suggest_float("momentum", 0., 0.95)

        # Training parameters
        params["epochs"] = trial.suggest_int("epochs", 25, 100)
        params["batch_size"] = trial.suggest_categorical("batch_size", [128, 256, 512])

        # End of Multi-Layer Perceptron

    elif model_name == "embed_lstm":

        # LSTM

        # Main parameters
        params["random_state"] = trial.suggest_categorical("random_state", [seed])

        # LSTM parameters
        params["bidirectional"] = trial.suggest_categorical("bidirectional", [True, False])
        params["nb_lstm_layers"] = trial.suggest_int("nb_lstm_layers", 1, 2)
        params["lstm_hidden_dim"] = trial.suggest_int("lstm_hidden_dim", 250, 350)
        params["lstm_dropout"] = trial.suggest_float("lstm_dropout", 0.25, 0.4)
        params["lstm_recurrent_dropout"] = trial.suggest_float("lstm_recurrent_dropout", 0.25, 0.4)

        # MLP parameters
        params["nb_mlp_layers"] = trial.suggest_int("nb_mlp_layers", 1, 2)
        params["mlp_hidden_dim"] = trial.suggest_int("mlp_hidden_dim", 400, 600)
        params["mlp_hidden_act"] = \
            trial.suggest_categorical("mlp_hidden_act", ["tanh", "relu"])
        params["mlp_hidden_init"] = \
            trial.suggest_categorical("mlp_hidden_init", ["uniform", "glorot_uniform"])
        params["mlp_dropout"] = trial.suggest_float("mlp_dropout", 0.7, 0.9)
        params["batch_norm"] = trial.suggest_categorical("batch_norm", [False])

        # Optimizer parameters
        params["learning_rate"] = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)
        params["optim_name"] = trial.suggest_categorical("optim_name", ["adam", "rmsprop"])
        if params["optim_name"] == "adam":
            params["beta_1"] = trial.suggest_categorical("beta_1", [0.9])
            params["beta_2"] = trial.suggest_categorical("beta_2", [0.999])
        elif params["optim_name"] == "rmsprop":
            params["rho"] = trial.suggest_float("rho", 0.8, 0.95)
            params["momentum"] = trial.suggest_float("momentum", 0., 0.95)

        # Training parameters
        params["epochs"] = trial.suggest_int("epochs", 25, 40)
        params["batch_size"] = trial.suggest_categorical("batch_size", [128, 256, 512])

        # End of LSTM

    elif model_name == "stacking":

        # Stacking

        params["random_state"] = trial.suggest_categorical("random_state", [seed])

        # End of Stacking

    else:

        err_msg = f"Unable to create optuna gridsearch for {model_name}."
        raise ValueError(err_msg)

    return params
