"""Initialize a model based on its name and parameters."""


from typing import Any, Dict, Optional, Union, Tuple

from tqdm import tqdm

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from keras.preprocessing import sequence, text


from preprocessing.features.embeddings import load_embeddings
from engine.models.base import BaseEstimator
from engine.models.deep import DeepEstimator


BASE_MODELS_NAMES = [
    "rfc", "etc", "xgboost", "lightgbm", "catboost", "logreg", "stacking"
]

DEEP_MODELS_NAMES = [
    "mlp", "embed_lstm"
]


def create_model(
    model_name: str, params: Dict[str, Any], embeddings_matrix: Optional[ndarray] = None
) -> Union[BaseEstimator, DeepEstimator]:
    """Create a model.

    Parameters
    ----------
    model_name : str
        The name of model following project usage. See README.md for more information about
        available models.

    params : dict of str
        A dictionary of parameters for chosen **model_name**.

    embeddings_matrix : optional ndarray, default=None
        Matrix of tokens to embeddings for embed models.

    Returns
    -------
    model : `BaseEstimator` or `DeepEstimator`
        Corresponding model from the catalogue.

    Raises
    ------
    ValueError
        If the **model_name** is not supported.
    """
    if model_name in BASE_MODELS_NAMES:

        model = BaseEstimator(model_name, params)

    elif model_name in DEEP_MODELS_NAMES:

        model = DeepEstimator(model_name, params, embeddings_matrix=embeddings_matrix)

    else:

        err_msg = f"Unknown model {model_name}."
        raise ValueError(err_msg)

    return model


def prepare_data(
    model_name: str, train_df: Union[DataFrame, ndarray], test_df: Union[DataFrame, ndarray],
    embeddings_name: str = ""
) -> Tuple[Union[DataFrame, ndarray], ...]:
    """Specific data transformation for models.

    Parameters
    ----------
    model_name : str
        The name of model following project usage. See README.md for more information about
        available models.

    train_df : DataFrame
        Training dataframe containing the final features and training labels.

    test_df : DataFrame
        Test dataframe containing the final features.

    embeddings_name : str, default=""
        Name of the embeddings type, which can be of "glove" or "google". If anything else, it will
        be computed with the training and additional documents.

    Returns
    -------
    x_train : DataFrame or ndarray
        Training features.

    x_test : DataFrame or ndarray
        Test features

    embeddings_matrix : ndarray
        Matrix of tokens to embeddings for embed models.
    """
    if "embed" in model_name:

        # Retrieve embeddings
        more_docs = pd.read_json("data/documents.json")

        embeddings_index = load_embeddings(
            train_df, more_docs=more_docs, embeddings_name=embeddings_name
        )

        # Get training and test summaries
        x_train = train_df["summary"].to_numpy()
        x_test = test_df["summary"].to_numpy()

        # Using keras tokenizer here
        token = text.Tokenizer(num_words=None)
        max_len = 70

        token.fit_on_texts(list(x_train))
        x_train = token.texts_to_sequences(x_train)
        x_test = token.texts_to_sequences(x_test)

        # Zero pad the sequences
        x_train = sequence.pad_sequences(x_train, maxlen=max_len)
        x_test = sequence.pad_sequences(x_test, maxlen=max_len)

        word_index = token.word_index

        # Create an embedding matrix for the words we have in the dataset
        embeddings_matrix = np.zeros((len(word_index) + 1, 300))

        for word, i in tqdm(word_index.items()):

            if word in embeddings_index:
                embeddings_matrix[i] = embeddings_index[word]
            else:
                np.random.normal(size=300)

    else:

        x_train = train_df
        x_test = test_df
        embeddings_matrix = np.empty(0)

    return x_train, x_test, embeddings_matrix
