"""Main function to create and save features."""


import argparse

from typing import List, Optional, Tuple

import warnings

import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler, StandardScaler


from preprocessing.reader import check_features_names
from preprocessing.features.embeddings import create_embed_feat
from preprocessing.features.gltr import create_gltr_feat
from preprocessing.features.polynomial import create_poly_feat
from preprocessing.features.manual_regex import create_regex_feat
from preprocessing.features.pos_tagging import create_tagging_feat
from preprocessing.features.tfidf import create_idf_feat
from preprocessing.reduction.droping import drop_corr_feat, drop_list_feat
from preprocessing.reduction.pca import perform_pca

from utils.args_fmt import float_zero_one


warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def create_features(
    seed: int = 42,
    regex_feat: Optional[List[str]] = None,
    idf_feat: Optional[List[str]] = None,
    delta: float = 0.1,
    b: float = 0.5,
    k_1: float = 1.,
    svd_components: int = 1,
    tag_feat: Optional[List[str]] = None,
    gltr_feat: Optional[List[str]] = None,
    embed_feat: Optional[List[str]] = None,
    fixed_poly_feat: Optional[List[str]] = None,
    poly_feat: Optional[List[str]] = None,
    all_poly_feat: bool = False,
    poly_degree: int = 2,
    excl_feat: Optional[List[str]] = None,
    corr_threshold: float = 1.,
    rescale_data: bool = True,
    scaling_method: str = "standard",
    pca_ratio: float = 1.,
    save_data: bool = True,
    file_suffix: str = "final"
) -> Tuple[DataFrame, ...]:
    """Retrieve local data files and perform preprocessing operations.

    Parameters
    ----------
    seed : int, default=42
        Seed to use everywhere for reproducibility.

    regex_feat : optional list of str, default=None
        List of all regex features to compute. The name of a regex feature should be of the form
        "A_B". A is the name of the regex expression, for example "upper_word". B is the type of
        feature we want, it can be "count" for the number of instances, "avg" for the average
        length of instances, "overlap" for the number of instances both found in the summary and
        the original document or "ratio" for the ratio between the number of instances found in the
        summary and the number of instances found in the document.

    idf_feat : optional list of str, default=None
        List of all tf-idf features names to compute. The name of an idf feature is expected to be
        of the form "A_B". A is a sequence of characters for the composition performed in
        `src.preprocessing.features.tfidf.idf_composition()`. B is the type of feature we want, it
        can be "count" for the term frequencies, "lda" for the latent dirichlet allocation, "idf"
        for idf features after a PCA transformation that keeps **svd_components** components from
        the initial counter features or "cos" for the cosine similarity between the summary and the
        original document.

    delta : float, default=0.1
        Parameter value for "d" composition in tf-idf.

    b : float, default=0.5
        Parameter value for "p" composition in tf-idf.

    k_1 : float, default=1.0
        Parameter value for "k" composition in tf-idf.

    svd_components : int, default=1
        Number of components for the truncated SVD component analysis in tf-idf.

    tag_feat : optional list of str, default=None
        List of all tagging features to compute. The name of a tagging feature must be of the form
        "A_B". A is a tag from all available tags in `nltk`. B is the type of feature we want, it
        can be "count" for the number of instances, "avg" for the average length of instances,
        "overlap" for the number of instances both found in the summary and the original document
        or "ratio" for the ratio between the number of instances found in the summary and the
        number of instances found in the document.

    gltr_feat : optional list of str, default=None
        List of all GLTR features to compute. The name of a GLTR feature should be of the form
        "A_B". A is the name of a topk value computed by GLTR, it can be "count" or "frac". B is
        the number of bins to compute the feature.

    embed_feat : optional list of str, default=None
        List of all embed features to compute. The name of a embed feature should be of the form
        "A". A is the name of an embedding from "glove" and "google", or if any other name, it
        will be trained.

    fixed_poly_feat : optional list of str, default=None
        List of specific polynomial features. A fixed polynomial feature must be of the form
        "A B C". A, B and C can be features or powers of features, and their product will be
        computed.

    poly_feat : optional list of str, default=None
        List of polynomial features to compute of which interaction terms will be computed.

    all_poly_feat : bool, default=False
        If True, will use all the computed features for polynomial interaction. Use with caution.

    poly_degree : int, default=2
        Define the degree until which products and powers of features are computed. If 1 or less,
        there will be no polynomial features.

    excl_feat : optional list of str, default=None
        List of features names to drop.

    corr_threshold : float, default=1.0
        Correlation threshold to select features.

    rescale_data : bool, default=True
        If True, will rescale all features with zero mean and unit variance.

    scaling_method : str, default="standard"
        If "standard", features are rescaled with zero mean and unit variance. If "positive",
        features are rescaled between zero and one.

    pca_ratio : float, default=1.0
        Variance ratio parameter for the Principal Component Analysis.

    save_data : bool, default=True
        If True, will save the computed features in two csv files, one for training and one for
        testing.

    file_suffix : str, default="final"
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

    # Compute regex features (example: "upper_word_count")
    create_regex_feat(train_df, test_df=test_df, regex_feat=regex_feat)

    # Compute idf features (example: "ldp_idf" or "kp_cos")
    create_idf_feat(
        seed, train_df, test_df=test_df, idf_feat=idf_feat, more_docs=documents, delta=delta, b=b,
        k_1=k_1, svd_components=svd_components
    )

    # Compute tagging features (example: "NN_count")
    create_tagging_feat(
        seed, train_df, test_df=test_df, tag_feat=tag_feat, more_docs=documents, delta=delta, b=b,
        k_1=k_1, svd_components=svd_components
    )

    # Compute GLTR features (example: "count_4")
    train_df, test_df = create_gltr_feat(train_df, test_df=test_df, gltr_feat=gltr_feat)

    # Compute embeddings features (example: "glove")
    create_embed_feat(train_df, test_df=test_df, embed_feat=embed_feat, more_docs=documents)

    # Compute polynomial (example: "char_count word_count") and interaction (all features)
    if all_poly_feat:
        fixed_poly_feat = []
        poly_feat = list(train_df.columns)
        poly_feat.remove("summary")
        poly_feat.remove("document")

    create_poly_feat(
        train_df, test_df=test_df, fixed_poly_feat=fixed_poly_feat, poly_feat=poly_feat,
        poly_degree=poly_degree
    )

    # Drop excluded features
    drop_list_feat(train_df, test_df=test_df, excl_feat=excl_feat)

    # Drop correlated features
    drop_corr_feat(train_df, test_df=test_df, corr_threshold=corr_threshold)

    # Rescale the data
    if rescale_data:

        if scaling_method == "standard":
            sc = StandardScaler()
        elif scaling_method == "positive":
            sc = MinMaxScaler()
        else:
            err_msg = f"Unsupported scaling method {scaling_method}."
            raise ValueError(err_msg)

        train_df[train_df.columns] = sc.fit_transform(train_df[train_df.columns])
        test_df[test_df.columns] = sc.transform(test_df[test_df.columns])

    elif not rescale_data and pca_ratio < 1.:

        warn_msg = "Warning: Rescaling data is recommended when performing PCA."
        warn_msg += " Set rescale_data to True or pca_ratio to 1."
        print(warn_msg)

    # Reduce the dimensionality with PCA
    train_df, test_df = perform_pca(seed, train_df, test_df=test_df, pca_ratio=pca_ratio)

    # Print some information
    print(f"\nFinal training shape: {train_df.shape}")
    print(f"Features names: {list(train_df.columns)}")

    # Re-merge the label variable
    train_df = train_df.merge(y_train, left_index=True, right_index=True, how="left")

    # Check features names and change them if needed
    check_features_names((train_df, test_df))

    # Save the data
    if save_data:
        train_df.to_csv(
            path_or_buf="data/train_" + file_suffix + ".csv", header=True, index_label="id"
        )
        test_df.to_csv(
            path_or_buf="data/test_" + file_suffix + ".csv", header=True, index_label="id"
        )

    return train_df, test_df


if __name__ == "__main__":

    # Command lines
    parser_desc = "Main file to prepare data and features."
    parser = argparse.ArgumentParser(description=parser_desc)

    # Seed
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="""
             Seed to use everywhere for reproducbility.
             Default: 42.
             """
    )

    # Regex features
    parser.add_argument(
        "--regex-feat",
        default=[
            "char_count", "char_ratio",
            "word_count", "word_overlap", "word_ratio",
            "sent_avg",
            "upper_word_ratio",
            "group_overlap",
            "is_last_ponct_count",
            "space_before_ponct_count",
            "word_3_count"
        ],
        nargs="*",
        help="""
             List of all regex features to compute. The name of a regex feature should be of the
             form "A_B". A is the name of the regex expression, for example "upper_word". B is the
             type of feature we want, it can be "count" for the number of instances, "avg" for the
             average length of instances, "overlap" for the number of instances both found in the
             summary and the original document or "ratio" for the ratio between the number of
             instances found in the summary and the number of instances found in the document.
             Example: --regex-feat char_count group_overlap.
             """
    )

    # Tf-idf features
    parser.add_argument(
        "--idf-feat",
        default=[
            "n_count", "n_idf", "n_lda"
        ],
        nargs="*",
        help="""
             List of all tf-idf features names to compute. The name of an idf feature is expected
             to be of the form "A_B". A is a sequence of characters for the composition performed
             in `src.preprocessing.features.tfidf.idf_composition()`. B is the type of feature we
             want, it can be "count" for the term frequencies, "lda" for the latent dirichlet
             allocation, "idf" for idf features after a PCA transformation that keeps
             **svd_components** components from the initial counter features or "cos" for the
             cosine similarity between the summary and the original document.
             """
    )

    parser.add_argument(
        "--idf-d",
        default=0.1,
        type=float,
        help="""
             Parameter value for "d" composition in tf-idf.
             Default: 0.1.
             """
    )
    parser.add_argument(
        "--idf-b",
        default=0.5,
        type=float,
        help="""
             Parameter value for "p" composition in tf-idf.
             Default: 0.5.
             """
    )
    parser.add_argument(
        "--idf-k",
        default=1.,
        type=float,
        help="""
             Parameter value for "k" composition in tf-idf.
             Default: 1.0.
             """
    )
    parser.add_argument(
        "--idf-svd",
        default=1,
        type=int,
        help="""
             Number of components for the truncated SVD component analysis in tf-idf.
             Default: 1.
             """
    )

    # Tagging features
    parser.add_argument(
        "--tag-feat",
        default=[
            "DT_count", "DT_overlap",
            "MD_count",
            "NN_overlap",
            "RB_count",
            "$_overlap",
            ":_overlap"
        ],
        nargs="*",
        help="""
             List of all tagging features to compute. The name of a tagging feature must be of the
             form "A_B". A is a tag from all available tags in `nltk` or "tags". B is the type of
             feature we want, it can be "count" for the number of instances, "avg" for the average
             length of instances, "overlap" for the number of instances both found in the summary
             and the original document, "ratio" for the ratio between the number of instances found
             in the summary and the number of instances found in the document or "C_D" where C and
             D are parameters for idf features.
             """
    )

    # GLTR features
    parser.add_argument(
        "--gltr-feat",
        default=[
            "count_4", "frac_10"
        ],
        nargs="*",
        help="""
             List of all GLTR features to compute. The name of a GLTR feature should be of the form
             "A_B". A is the name of a topk value computed by GLTR, it can be "count" or "frac". B
             is the number of bins to compute the feature.
             """
    )

    # Embeddings features
    parser.add_argument(
        "--embed-feat",
        default=[],
        nargs="*",
        help="""
             List of all embed features to compute. The name of a embed feature should be of the
             form "A". A is the name of an embedding from "glove" and "google", or if any other
             name, it will be trained.
             """
    )

    # Polynomial features
    parser.add_argument(
        "--fixed-poly-feat",
        default=[],
        nargs="*",
        help="""
             List of specific polynomial features. A fixed polynomial feature must be of the form
             "A B C". A, B and C can be features or powers of features, and their product will be
             computed.
             Example: --fixed-poly-feat "char_count^2 group_overlap".
             """
    )

    parser.add_argument(
        "--poly-feat",
        default=[],
        nargs="*",
        help="""
             List of polynomial features to compute of which interaction terms will be computed.
             Example: --poly-feat ldp_idf char_count.
             """
    )
    parser.add_argument(
        "--all-poly-feat",
        action="store_true",
        help="""
             Use this option to activate polynomial interaction of all features. Use with
             caution.
             Default: Deactivated.
             """
    )
    parser.set_defaults(all_poly_feat=False)
    parser.add_argument(
        "--poly-degree",
        default=2,
        type=int,
        help="""
             Define the degree until which products and powers of features are computed. If 1 or
             less, there will be no polynomial features.
             Default: 2.
             """
    )

    # Excluded features
    parser.add_argument(
        "--excl-feat",
        default=[
            "document", "summary",
            "n_lda_2", "n_lda_3"
        ] + [
            f"topk_frac_{i}" for i in range(1, 10)
        ],
        nargs="*",
        help="""
             List of features names to drop after computation.
             Example: --excl-feat "char_count^2 group_overlap".
             """
    )

    # Correlation threshold
    parser.add_argument(
        "--max-correlation",
        default=1.,
        type=float,
        help="""
             Correlation threshold to select features.
             Default: 1.0.
             """
    )

    # Rescale data
    parser.add_argument(
        "--rescale-data",
        action="store_true",
        help="""
             Use this option to activate rescaling the data sets.
             Default: Activated.
             """
    )
    parser.add_argument(
        "--no-rescale-data",
        action="store_false",
        dest="rescale-data",
        help="""
             Use this option to deactivate rescaling the data sets.
             Default: Activated.
             """
    )
    parser.set_defaults(rescale_data=True)

    parser.add_argument(
        "--scaling-method",
        default="standard",
        type=str,
        help="""
             If "standard", features are rescaled with zero mean and unit variance. If "positive",
             features are rescaled between zero and one.
             Default: "standard".
             """
    )

    # PCA ratio
    parser.add_argument(
        "--pca-ratio",
        default=1.,
        type=float,
        help="""
             Variance ratio parameter for the Principal Component Analysis.
             Default: 1.0.
             """
    )

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
        default="final",
        type=str,
        help="""
             Suffix to append to the training and test files if **save_data** is True.
             Default: "final".
             """
    )

    # End of command lines
    args = parser.parse_args()

    create_features(
        seed=args.seed,
        regex_feat=args.regex_feat,
        idf_feat=args.idf_feat,
        delta=args.idf_d,
        b=args.idf_b,
        k_1=args.idf_k,
        svd_components=args.idf_svd,
        tag_feat=args.tag_feat,
        gltr_feat=args.gltr_feat,
        embed_feat=args.embed_feat,
        fixed_poly_feat=args.fixed_poly_feat,
        poly_feat=args.poly_feat,
        all_poly_feat=args.all_poly_feat,
        poly_degree=args.poly_degree,
        excl_feat=args.excl_feat,
        corr_threshold=float_zero_one(args.max_correlation),
        rescale_data=args.rescale_data,
        scaling_method=args.scaling_method,
        pca_ratio=float_zero_one(args.save_data),
        save_data=args.save_data,
        file_suffix=args.file_suffix
    )
