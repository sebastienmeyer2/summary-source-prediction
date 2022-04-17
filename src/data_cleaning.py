"""Functions to clean documents and summaries before using pretrained models."""


import argparse

from typing import Tuple

import re

import warnings

from tqdm import tqdm

import pandas as pd
from pandas import DataFrame


warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

tqdm.pandas()


def apply_clean_text(text: str) -> str:
    """Remove extra white spaces in a text.

    Parameters
    ----------
    text : str
        Initial text to clean.

    Returns
    -------
    text : str
        Cleaned text.
    """
    text = re.sub(r"\s([?.!,:;'](?:\s|$))", r"\1", text)  # remove space before ponctuation
    text = re.sub(" +", " ", text)  # strip extra white space
    text = text.strip()  # strip leading and trailing white space

    return text


def create_clean_summary(
    clean_text: bool = True, save_data: bool = True, file_suffix: str = "clean"
) -> Tuple[DataFrame, ...]:
    """Retrieve local data files and perform preprocessing operations.

    Parameters
    ----------
    save_data : bool, default=True
        If True, will save the computed features in two csv files, one for training and one for
        testing.

    file_suffix : str, default="clean"
        Suffix to append to the training and test files if **save_data** is True.

    Returns
    -------
    train_df : DataFrame
        Training dataframe containing the final features and training labels.

    test_df : DataFrame
        Test dataframe containing the final features.

    Raises
    ------
    ValueError
        If **scaling_method** is not supported.
    """
    # Read files
    train_df = pd.read_json("data/train_set.json")
    test_df = pd.read_json("data/test_set.json")
    documents = pd.read_json("data/documents.json")

    # Extract target (it will be merged back later)
    label_var = ["label"]
    y_train = train_df[label_var]
    train_df.drop(columns=label_var, inplace=True)

    # Clean summaries
    if clean_text:

        train_df["document_clean"] = train_df["document"].progress_apply(apply_clean_text)
        train_df["summary_clean"] = train_df["summary"].progress_apply(apply_clean_text)
        test_df["document_clean"] = test_df["document"].progress_apply(apply_clean_text)
        test_df["summary_clean"] = test_df["summary"].progress_apply(apply_clean_text)
        documents["document_clean"] = documents["document"].progress_apply(apply_clean_text)

    # Re-merge the label variable
    train_df = train_df.merge(y_train, left_index=True, right_index=True, how="left")

    # Save the data
    if save_data:

        train_df.to_csv(
            path_or_buf="data/train_" + file_suffix + ".csv", header=True, index_label="id"
        )
        test_df.to_csv(
            path_or_buf="data/test_" + file_suffix + ".csv", header=True, index_label="id"
        )
        documents.to_csv(
            path_or_buf="data/documents_" + file_suffix + ".csv", header=True, index_label="id"
        )

    return train_df, test_df


if __name__ == "__main__":

    # Command lines
    parser_desc = "Main file to clean data for embedded models."
    parser = argparse.ArgumentParser(description=parser_desc)

    # Clean text
    parser.add_argument(
        "--clean-text",
        action="store_true",
        help="""
             Use this option to activate the application of a slight cleaning of the data, that is,
             removing extra white spaces.
             Default: Activated.
             """
    )
    parser.add_argument(
        "--no-clean-text",
        action="store_true",
        help="""
             Use this option to deactivate the application of a slight cleaning of the data, that
             is, removing extra white spaces.
             Default: Activated.
             """
    )
    parser.set_defaults(clean_text=True)

    # Save data
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="""
             Use this option to activate saving the data sets.
             Default: Activated.
             """
    )
    parser.add_argument(
        "--no-save-data",
        action="store_false",
        dest="save-data",
        help="""
             Use this option to deactivate saving the data sets.
             Default: Activated.
             """
    )
    parser.set_defaults(save_data=True)

    parser.add_argument(
        "--file-suffix",
        default="clean",
        type=str,
        help="""
             Suffix to append to the training and test files if **save_data** is True.
             Default: "clean".
             """
    )

    # End of command lines
    args = parser.parse_args()

    create_clean_summary(
        clean_text=args.clean_text, save_data=args.save_data, file_suffix=args.file_suffix
    )
