"""Gather utilitary functions to read the data."""


from typing import Tuple

import pandas as pd
from pandas import DataFrame


SPECIAL_NLTK_TAGS = {
    "$": "dollar_tag",
    "''": "apostrophe_tag",
    "(": "obracket_tag",
    ")": "cbracket_tag",
    ",": "comma_tag",
    "--": "dash_tag",
    ".": "period_tag",
    ":": "column_tag",
    "``": "code_tag"
}


def get_data(data_path: str = "data/", file_suffix: str = "final") -> Tuple[DataFrame, ...]:
    """Retrieve local data files and perform preprocessing oprations.

    Parameters
    ----------
    data_path : str, default="data/"
        Path to the data folder.

    file_suffix : str, default="final"
        Suffix to the training and test filenames.

    Returns
    -------
    train_df : DataFrame
        Training features.

    y_train : DataFrame
        Training labels.

    test_df : DataFrame
        Test features.
    """
    # Read files
    path_to_train = data_path + "train_" + file_suffix + ".csv"
    path_to_test = data_path + "test_" + file_suffix + ".csv"

    train_df = pd.read_csv(path_to_train, index_col=["id"])
    test_df = pd.read_csv(path_to_test, index_col=["id"])

    # Check columns names
    check_features_names((train_df, test_df))

    # Extract variables
    label_var = ["label"]
    y_train = train_df[label_var]
    train_df = train_df.drop(columns=label_var)
    test_df = test_df.drop(columns=label_var, errors="ignore")

    return train_df, y_train, test_df


def check_features_names(dfs: Tuple[DataFrame, ...]):
    """Check and change features names to remove special symbols.

    This is particularly useful for some machine learning modules which do not accept features
    names with special symbols, such as `lightgbm`.

    Parameters
    ----------
    dfs : tuple of DataFrame
        All dataframes to check.
    """
    for df in dfs:

        new_cols = {k: k for k in list(df.columns)}

        for old_col in new_cols.keys():

            for tag, rep_tag in SPECIAL_NLTK_TAGS.items():

                if tag in old_col:

                    new_cols[old_col] = old_col.replace(tag, rep_tag)

        df.rename(columns=new_cols, inplace=True)
