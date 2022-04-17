"""Wrapper for deep models."""


from typing import Any, Dict, Optional, Union

import numpy as np
from numpy import ndarray
from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.layers import Dense, Dropout, BatchNormalization, Embedding, Bidirectional, LSTM
from keras.utils import np_utils
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop  # pylint: disable=no-name-in-module


class DeepEstimator():
    """A wrapper for deep learning models."""

    def __init__(
        self, model_name: str, params: Dict[str, Any], embeddings_matrix: Optional[ndarray] = None
    ):
        """The constructor of the class.

        Parameters
        ----------
        model_name : str
            The name of model following project usage. See README.md for more information.

        params : dict of {str: any}
            A dictionary of parameters for chosen **model_name**. It contains all parameters to
            initialize and fit the model.

        embeddings_matrix : optional ndarray, default=None
            Matrix of tokens to embeddings for embed models.
        """
        self.model_name = model_name

        # Model parameters
        self.model_params = params
        self.embeddings_matrix = embeddings_matrix

        # Label prediction
        if model_name == "mlp":
            self.model = create_mlp(self.model_params)
        elif model_name == "embed_lstm":
            self.model = create_embed_lstm(self.model_params, embeddings_matrix=embeddings_matrix)
        else:
            raise ValueError(f"{model_name} is not implemented. Check available models.")

        # Compile the model (transparent if not tf.keras.Model)
        self.compile()

    def compile(self):
        """Compile contained model."""
        optimizer = create_optimizer(self.model_params)

        self.model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

    def fit(self, x_train: Union[ndarray, DataFrame], y_train: Union[ndarray, DataFrame]):
        """Wrapper of fit method.

        Parameters
        ----------
        x_train : ndarray or DataFrame
            Training features.

        y_train : ndarray or DataFrame
            Training labels.
        """
        # Get parameters
        seed = self.model_params.get("random_state", 42)
        batch_size = self.model_params.get("batch_size", 512)
        epochs = self.model_params.get("epochs", 30)
        verbose = self.model_params.get("verbose", 1)

        # The neural network needs one-hot encoded labels in numpy
        if isinstance(x_train, DataFrame):
            x_train = x_train.to_numpy()

        y_train_hot = np_utils.to_categorical(y_train)

        # Callbacks
        def steps_decay(lr0: float, s: int):
            def steps_decay_fn(epoch: int):
                return 0.1**(int(epoch/s)) * lr0
            return steps_decay_fn

        earlystop = EarlyStopping(
            monitor="val_loss", min_delta=0, patience=3, verbose=0, mode="auto"
        )

        decay_fn = steps_decay(self.model_params.get("learning_rate", 0.001), 40)
        lr_scheduler = LearningRateScheduler(decay_fn)

        # Create an eval set
        x_train, x_eval, y_train_hot, y_eval_hot = train_test_split(
            x_train, y_train_hot, random_state=seed
        )

        # Fit the nn
        self.model.fit(
            x_train, y_train_hot, epochs=epochs, batch_size=batch_size, verbose=verbose,
            callbacks=[earlystop, lr_scheduler], validation_data=(x_eval, y_eval_hot)
        )

    def predict(self, x_eval: Union[ndarray, DataFrame]) -> Union[ndarray, DataFrame]:
        """Wrapper of predict method.

        Parameters
        ----------
        x_eval : ndarray or DataFrame
            Evaluation features.

        Returns
        -------
        y_pred : ndarray
            Predicted labels on evaluation set.
        """
        if isinstance(x_eval, DataFrame):
            x_eval = x_eval.to_numpy()

        y_pred = np.argmax(self.model.predict(x_eval), axis=1)

        return y_pred

    def predict_proba(self, x_eval: Union[ndarray, DataFrame]) -> Union[ndarray, DataFrame]:
        """Wrapper of predict_proba method.

        Parameters
        ----------
        x_eval : ndarray or DataFrame
            Evaluation features.

        Returns
        -------
        y_pred_probs : ndarray
            Predicted probabilities on evaluation set.
        """
        y_pred_softmax = self.model.predict(x_eval)

        # Rescale to get probabilities
        sc = MinMaxScaler(feature_range=(0, 1))
        y_pred_probs = sc.fit_transform(y_pred_softmax)

        return y_pred_probs


def create_mlp(params: Dict[str, Any]) -> Sequential:
    """Build up a feed-forward network or multi-layer perceptron.

    Parameters
    ----------
    params : dict of {str: any}
        A dictionary of parameters for the feed-forward network. It contains all parameters to
        initialize and fit the model.

    Returns
    -------
    model : `Sequential`
        The complete sequence of layers for the feed-forward network.
    """
    # Get parameters
    seed = params.get("random_state", 42)
    nb_hidden_layers = params.get("nb_hidden_layers", 1)
    input_dim = params.get("input_dim", 300)
    hidden_dim = params.get("hidden_dim", 150)
    hidden_act = params.get("hidden_act", "sigmoid")
    hidden_init = params.get("hidden_init", "uniform")
    dropout = params.get("dropout", 0.)
    batch_norm = params.get("batch_norm", False)

    # Build the model
    model = Sequential()

    for i in range(nb_hidden_layers):

        if i == 0:

            model.add(
                Dense(
                    hidden_dim, input_dim=input_dim, activation=hidden_act,
                    kernel_initializer=hidden_init
                )
            )

        else:

            model.add(
                Dense(hidden_dim, activation=hidden_act, kernel_initializer=hidden_init)
            )

        if dropout > 0.:

            model.add(Dropout(dropout, seed=seed))

        if batch_norm:

            model.add(BatchNormalization())

    model.add(Dense(2, activation="softmax", kernel_initializer=hidden_init))

    return model


def create_embed_lstm(
    params: Dict[str, Any], embeddings_matrix: Optional[ndarray] = None
) -> Sequential:
    """Build up a LSTM with embeddings.

    Parameters
    ----------
    params : dict of {str: any}
        A dictionary of parameters for the feed-forward network. It contains all parameters to
        initialize and fit the model.

    embeddings_matrix : optional ndarray, default=None
        Matrix of tokens to embeddings for embed models.

    Returns
    -------
    model : `Sequential`
        The complete sequence of layers for the feed-forward network.

    Raises
    ------
    ValueError
        If no embeddings are passed.
    """
    # Get parameters
    seed = params.get("seed", 42)
    input_dim = params.get("input_dim", 300)

    model = Sequential()

    # Build the embeddings
    input_length = params.get("max_len", 70)

    if embeddings_matrix is not None:

        model.add(
            Embedding(
                embeddings_matrix.shape[0], 300, weights=[embeddings_matrix],
                input_length=input_length, trainable=False
            )
        )

    else:

        raise ValueError("No embeddings for embedded model!")

    # Build the LSTM
    bidirectional = params.get("bidirectional", False)
    nb_lstm_layers = params.get("nb_lstm_layers", 1)
    lstm_hidden_dim = params.get("lstm_hidden_dim", 300)
    lstm_dropout = params.get("lstm_dropout", 0.3)
    lstm_recurrent_dropout = params.get("lstm_recurrent_dropout", 0.3)

    for i in range(nb_lstm_layers - 1):

        if bidirectional:

            model.add(
                Bidirectional(
                    LSTM(
                        lstm_hidden_dim, dropout=lstm_dropout,
                        recurrent_dropout=lstm_recurrent_dropout, return_sequences=True
                    )
                )
            )

        else:

            model.add(
                LSTM(
                    lstm_hidden_dim, dropout=lstm_dropout,
                    recurrent_dropout=lstm_recurrent_dropout, return_sequences=True
                )
            )

    if bidirectional:

        model.add(
            Bidirectional(
                LSTM(
                    lstm_hidden_dim, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout
                )
            )
        )

    else:

        model.add(
            LSTM(
                lstm_hidden_dim, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout
            )
        )

    # Build the feed-forward network
    nb_mlp_layers = params.get("nb_mlp_layers", 2)
    mlp_hidden_dim = params.get("mlp_hidden_dim", 1024)
    mlp_hidden_act = params.get("mlp_hidden_act", "relu")
    hidden_init = params.get("hidden_init", "glorot_uniform")
    mlp_dropout = params.get("mlp_dropout", 0.8)
    batch_norm = params.get("batch_norm", False)

    for i in range(nb_mlp_layers):

        if i == 0:

            model.add(
                Dense(
                    mlp_hidden_dim, input_dim=input_dim, activation=mlp_hidden_act,
                    kernel_initializer=hidden_init
                )
            )

        else:

            model.add(
                Dense(mlp_hidden_dim, activation=mlp_hidden_act, kernel_initializer=hidden_init)
            )

        if mlp_dropout > 0.:

            model.add(Dropout(mlp_dropout, seed=seed))

        if batch_norm:

            model.add(BatchNormalization())

    model.add(Dense(2, activation="softmax", kernel_initializer=hidden_init))

    return model


def create_optimizer(params: Dict[str, Any]) -> Union[Adam, RMSprop]:
    """Build up an optimizer.

    Parameters
    ----------
    params : dict of {str: any}
        A dictionary of parameters for the model. It contains all parameters to initialize and fit
        the model.

    Returns
    -------
    optimizer : `Adam` or `RMSProp`
        Initialized optimizer.

    Raises
    ------
    ValueError
        If the optimizer is not supported.
    """
    optim_name = params.get("optim_name", "adam")

    if optim_name == "adam":

        learning_rate = params.get("learning_rate", 0.001)
        beta_1 = params.get("beta_1", 0.9)
        beta_2 = params.get("beta_2", 0.999)

        optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

    elif optim_name == "rmsprop":

        learning_rate = params.get("learning_rate", 0.001)
        rho = params.get("rho", 0.9)
        momentum = params.get("momentum", 0.)

        optimizer = RMSprop(learning_rate=learning_rate, rho=rho, momentum=momentum)

    else:

        err_msg = f"Unknown optimizer {optim_name}."
        raise ValueError(err_msg)

    return optimizer
