"""Compute embeddings features."""


from typing import Dict, List, Optional

import re
import string

from tqdm import tqdm

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

import nltk

import gensim
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import Word2Vec
from gensim.utils import effective_n_jobs


def text_to_tokens(
    text: str, stopwords: List[str], punct: str, remove_stopwords: bool = True
) -> List[str]:
    """Convert a sentence to a list of tokens.

    Parameters
    ----------
    text : str
        Text to convert.

    stopwords : list of str
        Stop words to remove from the **text** if **remove_stopwords** is True.

    punct : str
        All ponctuation symbols contained in a single string.

    remove_stopwords : bool, default=True
        If True, will remove the **stopwords** from **text**.

    Returns
    -------
    tokens : list of str
        Initial **text** converted to tokens.
    """
    # Step 1: Conversion to lower case
    text = str(text).lower()

    # Step 2: Punctuation removal
    text = "".join(char for char in text if char not in punct)  # preserving intra-word dashes
    text = re.sub(" +", " ", text)  # strip extra white space
    text = text.strip()  # strip leading and trailing white space

    # Step 3: Tokenization
    tokens = text.split(" ")

    # Step 4: Stopwords removal
    if remove_stopwords:
        tokens = [token for token in tokens if token not in stopwords]

    # Step 5: Alphanumeric removal
    tokens = [token for token in tokens if token.isalpha()]

    return tokens


def text_to_vec(
    embeddings_index: Dict[str, ndarray], text: str, stopwords: List[str], punct: str,
    remove_stopwords: bool = True
) -> ndarray:
    """Convert a sentence to an embedding.

    Parameters
    ----------
    embeddings_index: dict of {str: ndarray}
        A dictionary mapping tokens to their embeddings.

    text : str
        Text to convert.

    stopwords : list of str
        Stop words to remove from the **text** if **remove_stopwords** is True.

    punct : str
        All ponctuation symbols contained in a single string.

    remove_stopwords : bool, default=True
        If True, will remove the **stopwords** from **text**.

    Returns
    -------
    v : ndarray
        Rescaled total embedding of the **text**.
    """
    # Step 1: Conversion to lower case
    text = str(text).lower()

    # Step 2: Punctuation removal
    text = "".join(char for char in text if char not in punct)  # preserving intra-word dashes
    text = re.sub(" +", " ", text)  # strip extra white space
    text = text.strip()  # strip leading and trailing white space

    # Step 3: Tokenization
    tokens = text.split(" ")

    # Step 4: Stopwords removal
    if remove_stopwords:
        tokens = [token for token in tokens if token not in stopwords]

    # Step 5: Alphanumeric removal
    tokens = [token for token in tokens if token.isalpha()]

    # Step 6: Embeddings of all tokens
    M = []

    for token in tokens:

        if token in embeddings_index:

            M.append(embeddings_index[token])

    M = np.array(M)

    # Step 7: Embedding of the text
    v = M.sum(axis=0)

    if isinstance(v, ndarray):

        v = v / np.sqrt((v ** 2).sum())

    else:

        v = np.zeros(300)

    return v


def load_embeddings(
    train_df: DataFrame, more_docs: Optional[DataFrame] = None, embeddings_name: str = "glove"
) -> Dict[str, ndarray]:
    """Load pretrained embeddings and train new ones.

    Parameters
    ----------
    train_df : DataFrame
        Training dataframe containing "summary" and "document" columns.

    more_docs : optional DataFrame, default=None
        Additional dataframe containing a "document" column to use for idf computation.

    embeddings_name : str, default="glove"
        Name of the file where embeddings are stored. If unknown, will train a new Word2Vec model.

    Returns
    -------
    embeddings_index: dict of {str: ndarray}
        A dictionary mapping tokens to their embeddings.
    """
    # Retrieve embeddings
    print("\nCreating embeddings...")

    embeddings_index = {}

    if embeddings_name == "glove":

        embeddings_index = {}

        f = open("data/embed/glove_300.txt", encoding="utf8")

        for line in tqdm(f):

            values = line.strip().split(" ")
            try:
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
            except ValueError:
                print(values[0])
            embeddings_index[word] = coefs

        f.close()

        print(f"Found {len(embeddings_index)} word vectors.")

    elif embeddings_name == "google":

        embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(
            "data/embed/google_300.gz", binary=True
        )

    else:

        # Tokenization parameters
        stopwords = nltk.corpus.stopwords.words("english")
        punct = string.punctuation.replace("-", "")

        print("Cleaning documents...")

        train_df["document_token"] = train_df["document"].progress_apply(
            lambda x: text_to_tokens(x, stopwords, punct, remove_stopwords=True)
        )

        if more_docs is not None:

            more_docs["document_token"] = more_docs["document"].progress_apply(
                lambda x: text_to_tokens(x, stopwords, punct, remove_stopwords=True)
            )

        print("All documents clean.")

        if more_docs is None:
            all_docs = train_df["document_clean"].to_list()
        else:
            all_docs = train_df["document_token"].to_list() + more_docs["document_token"].to_list()

        class callback(CallbackAny2Vec):
            """Callback to print loss after each epoch."""
            def __init__(self):

                self.epoch = 0
                self.loss_to_be_subed = 0

            def on_epoch_end(self, model):

                total_loss = model.get_latest_training_loss()
                current_loss = total_loss - self.loss_to_be_subed
                self.loss_to_be_subed = total_loss

                print(f"Loss after epoch {self.epoch}: {current_loss}")

                self.epoch += 1

        w2v = Word2Vec(
            all_docs, vector_size=300, window=20, min_count=5, workers=effective_n_jobs(-1),
            epochs=25, compute_loss=True, callbacks=[callback()]
        )

        embeddings_index = w2v.wv

    print("Embeddings done.")

    return embeddings_index


def create_embed_feat(
    train_df: DataFrame, test_df: Optional[DataFrame] = None,
    embed_feat: Optional[List[str]] = None, more_docs: Optional[DataFrame] = None,
):
    """Auxiliary function to create embeddings features.

    Parameters
    ----------
    train_df : DataFrame
        Training dataframe containing "summary" and "document" columns.

    test_df : optional DataFrame, default=None
        Test dataframe containing "summary" and "document" columns.

    embed_feat : optional list of str, default=None
        List of all embed features to compute. The name of a embed feature should be of the form
        "A". A is the name of an embedding from "glove" and "google", or if any other name, it
        will be trained.

    more_docs : optional DataFrame, default=None
        Additional dataframe containing a "document" column to use for idf computation.
    """
    if embed_feat is None or len(embed_feat) == 0:
        return

    print("\nComputing embeddings features...")

    init_feat = set(train_df.columns)
    new_feat = set()

    # Tokenization parameters
    stopwords = nltk.corpus.stopwords.words("english")
    punct = string.punctuation.replace("-", "")

    for feat_name in embed_feat:

        embeddings_index = load_embeddings(
            train_df, more_docs=more_docs, embeddings_name=feat_name
        )

        # pylint: disable=cell-var-from-loop
        # Create embeddings vectors
        train_df["summary_" + feat_name] = train_df["summary"].progress_apply(
            lambda x: text_to_vec(embeddings_index, x, stopwords, punct, remove_stopwords=True)
        )
        if test_df is not None:
            test_df["summary_" + feat_name] = test_df["summary"].progress_apply(
                lambda x: text_to_vec(embeddings_index, x, stopwords, punct, remove_stopwords=True)
            )

        embed_cols = [f"summary_{feat_name}_{i}" for i in range(300)]

        train_df[embed_cols] = pd.DataFrame(
            train_df["summary_" + feat_name].tolist(), index=train_df.index
        )
        if test_df is not None:
            test_df[embed_cols] = pd.DataFrame(
                test_df["summary_" + feat_name].tolist(), index=test_df.index
            )

        new_feat = new_feat.union(embed_cols)

    # Drop intermediary features
    inter_feat = set(train_df.columns).difference(init_feat.union(new_feat))

    train_df.drop(columns=inter_feat, inplace=True)
    if test_df is not None:
        test_df.drop(columns=inter_feat, inplace=True)

    print(f"Number of embeddings features: {len(new_feat)}.")
