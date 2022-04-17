"""Compute features related to tf-idf."""


from typing import List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from scipy.sparse import coo_matrix

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity


def idf_composition(
    feat_prefix: str, counts: List[coo_matrix], d: int, avdl: float, delta: float = 0.1,
    b: float = 0.5, k_1: float = 1.
) -> List[coo_matrix]:
    """Compute the composition of different regularizations for tf-idf features.

    Parameters
    ----------
    feat_prefix : str
        Prefix of the feature name, which should contain the different composition to apply
        in correct order. Available compositions are "n", "w", "l", "d", "p" and "k".

    counts : list of coo_matrix
        List of sparse counts matrices on train and test dataframes.

    d : int
        Total number of documents that were used during the training of the counter, used for
        "p" composition in tf-idf.

    avdl : float
        Average length of documents that were used during the training of the counter, used for
        "p" composition in tf-df.

    delta : float, default=0.1
        Parameter value for "d" composition in tf-idf.

    b : float, default=0.5
        Parameter value for "p" composition in tf-idf.

    k_1 : float, default=1.0
        Parameter value for "k" composition in tf-idf.

    Returns
    -------
    copies : list of coo_matrix
        List of composed sparse counts matrices on train and test dataframes.

    Raises
    ------
    ValueError
        If the **feat_prefix** composition character is not supported.
    """
    copies = [cnt.copy() for cnt in counts]

    for c in feat_prefix:

        if c == "n":
            continue

        elif c == "w":
            for mtrx in copies:
                mtrx.data = 1 + np.log(mtrx.data)

        elif c == "l":
            for mtrx in copies:
                mtrx.data = 1 + np.log(1 + np.log(mtrx.data))

        elif c == "d":
            for mtrx in copies:
                mtrx.data = mtrx.data + delta

        elif c == "p":
            for mtrx in copies:
                mtrx.data = mtrx.data / (1 - b + b * (d/avdl))

        elif c == "k":
            for mtrx in copies:
                mtrx.data = ((k_1 + 1) * mtrx.data) / (k_1 + mtrx.data)

        else:

            err_msg = f"Unsupported composition {c} for idf feature."
            raise ValueError(err_msg)

    return copies


def create_idf_feat(
    seed: int, train_df: DataFrame, test_df: Optional[DataFrame] = None,
    idf_feat: Optional[List[str]] = None, more_docs: Optional[DataFrame] = None,
    max_features: int = 10000, delta: float = 0.1, b: float = 0.5, k_1: float = 1.,
    svd_components: int = 1, suffix: str = ""
):
    """Auxiliary function to create tf-idf features.

    Parameters
    ----------
    seed : int
        Seed to use everywhere for reproducibility.

    train_df : DataFrame
        Training dataframe containing "summary" and "document" columns.

    test_df : optional DataFrame, default=None
        Test dataframe containing "summary" and "document" columns.

    idf_feat : optional list of str, default=None
        List of all tf-idf features names to compute. The name of an idf feature is expected to be
        of the form "A_B". A is a sequence of characters for the composition performed in
        `src.preprocessing.features.tfidf.idf_composition()`. B is the type of feature we want, it
        can be "count" for the term frequencies, "lda" for the latent dirichlet allocation, "idf"
        for idf features after a PCA transformation that keeps **svd_components** components from
        the initial counter features or "cos" for the cosine similarity between the summary and the
        original document.

    more_docs : optional DataFrame, default=None
        Additional dataframe containing a "document" column to use for idf computation.

    max_features : int, default=10000
        Maximum number of features in the counter.

    delta : float, default=0.1
        Parameter value for "d" composition in tf-idf.

    b : float, default=0.5
        Parameter value for "p" composition in tf-idf.

    k_1 : float, default=1.0
        Parameter value for "k" composition in tf-idf.

    svd_components : int, default=1
        Number of components for the truncated SVD component analysis. This number must be smaller
        than the number of features in the counter.

    suffix : str, default=""
        Suffix for the summary and document columns for specific features such as counter on tags.
    """
    if idf_feat is None or len(idf_feat) == 0:
        return

    sum_col = "summary" + suffix
    doc_col = "document" + suffix

    print("\nComputing idf features...")

    init_feat = set(train_df.columns)
    new_feat = set()

    if more_docs is not None:
        all_docs = pd.concat((train_df[doc_col], more_docs[doc_col]))
    else:
        all_docs = train_df[doc_col]

    # Step 1: Count term frequencies on all documents and summaries
    ctr = CountVectorizer(max_features=max_features)
    ctr.fit(all_docs)  # train only on human-written documents
    train_d_cnt = ctr.transform(all_docs)
    train_s_cnt = ctr.transform(train_df[sum_col])
    if test_df is not None:
        test_d_cnt = ctr.transform(test_df[doc_col])
        test_s_cnt = ctr.transform(test_df[sum_col])

    # Step 2: Compute the inverse document frequency and extract the diagonal matrix
    t = TfidfTransformer(smooth_idf=False)
    t.fit(train_d_cnt)
    idf = t._idf_diag  # pylint: disable=protected-access

    # Step 3: Compute the different features
    counts = [train_d_cnt, train_s_cnt]
    if test_df is not None:
        counts.extend([test_d_cnt, test_s_cnt])

    d = len(all_docs)
    avdl = all_docs.str.len().mean()

    for feat_name in idf_feat:

        feat_split = feat_name.split("_")
        feat_prefix = feat_split[0]
        feat_type = feat_split[-1]

        # Step 3.1: Compose the term frequency matrices
        copies = idf_composition(feat_prefix, counts, d, avdl, delta=delta, b=b, k_1=k_1)

        # Step 3.2: Compute the tf-idf values
        copies_idf = [mtrx * idf for mtrx in copies]

        if feat_type == "count":

            svd = TruncatedSVD(n_components=svd_components, random_state=seed)
            svd.fit(copies[0])

            svd_col = [feat_name + suffix + f"_PCA_{i}" for i in range(1, svd_components + 1)]

            train_df[svd_col] = pd.DataFrame(svd.transform(copies[1]), index=train_df.index)
            if test_df is not None:
                test_df[svd_col] = pd.DataFrame(svd.transform(copies[3]), index=test_df.index)

            new_feat = new_feat.union(svd_col)

        elif feat_type == "lda":

            lda = LatentDirichletAllocation(n_components=3, random_state=seed, n_jobs=-1)

            lda.fit(copies[0])

            lda_col = [feat_name + suffix + f"_{i}" for i in range(3)]

            train_df[lda_col] = pd.DataFrame(lda.transform(copies[1]), index=train_df.index)
            if test_df is not None:
                test_df[lda_col] = pd.DataFrame(lda.transform(copies[3]), index=test_df.index)

            new_feat = new_feat.union(lda_col)

        elif feat_type == "idf":

            svd = TruncatedSVD(n_components=svd_components, random_state=seed)
            svd.fit(copies_idf[0])

            svd_col = [feat_name + suffix + f"_PCA_{i}" for i in range(1, svd_components + 1)]

            train_df[svd_col] = pd.DataFrame(svd.transform(copies_idf[1]), index=train_df.index)
            if test_df is not None:
                test_df[svd_col] = pd.DataFrame(svd.transform(copies_idf[3]), index=test_df.index)

            new_feat = new_feat.union(svd_col)

        elif feat_type == "cos":

            train_df[feat_name + suffix] = [
                cosine_similarity(copies_idf[0][i:i+1], copies_idf[1][i:i+1]).item()
                for i in range(len(train_df))
            ]
            if test_df is not None:
                test_df[feat_name + suffix] = [
                    cosine_similarity(copies_idf[2][i:i+1], copies_idf[3][i:i+1]).item()
                    for i in range(len(test_df))
                ]

            new_feat.add(feat_name + suffix)

        else:

            err_msg = f"From idf feature {feat_name}, unsupported feature type {feat_type}."
            raise ValueError(err_msg)

    # Drop intermediary features
    inter_feat = set(train_df.columns).difference(init_feat.union(new_feat))

    train_df.drop(columns=inter_feat, inplace=True)
    if test_df is not None:
        test_df.drop(columns=inter_feat, inplace=True)

    print(f"Number of idf features: {len(new_feat)}.")
