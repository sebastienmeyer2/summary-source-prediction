"""Compute features based on Part-of-Speech tagging."""


from typing import List, Optional, Set, Tuple

import re
import string

from pandas import DataFrame

from nltk import pos_tag
from nltk.corpus import stopwords as stopwords_collection
from nltk.stem import PorterStemmer


from preprocessing.features.tfidf import create_idf_feat


ALL_NLTK_TAGS = [
    "$", "''", "(", ")", ",", "--", ".", ":", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR",
    "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS",
    "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB",
    "``"
]


def text_to_tagged_tokens(
    text: str, stopwords: List[str], punct: str, remove_stopwords: bool = True,
    stemming: bool = True, pos_filtering: bool = True
) -> List[Tuple[str, str]]:
    """Convert a text to tagged tokens.

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

    stemming : bool, default=True
        If True, will stem the remaining words before tagging.

    pos_filtering : bool, default=True
        If True, will remove some tags from the **text**.

    Returns
    -------
    tagged_tokens : list of tuple of str
        Initial **text** converted to accepted tags with corresponding words in tuples.
    """
    # Step 1: Conversion to lower case
    text = text.lower()

    # Step 2: Punctuation removal
    text = "".join(char for char in text if char not in punct)  # preserving intra-word dashes
    text = re.sub(" +", " ", text)  # strip extra white space
    text = text.strip()  # strip leading and trailing white space

    # Step 3: Tokenization
    tokens = text.split(" ")

    # Step 4: Stopwords removal
    if remove_stopwords:
        tokens = [token for token in tokens if token not in stopwords]

    # Step 5: Stemming
    if stemming:
        stemmer = PorterStemmer()
        tokens_stemmed = []
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed

    # Step 6: Tagging
    tagged_tokens = pos_tag(tokens)

    # Step 7: Part-of-Speech based filtering
    if pos_filtering:

        tagged_tokens_keep = []
        for item in tagged_tokens:
            if (
                item[1] == "NN" or
                item[1] == "NNS" or
                item[1] == "NNP" or
                item[1] == "NNPS" or
                item[1] == "JJ" or
                item[1] == "JJS" or
                item[1] == "JJR"
            ):
                tagged_tokens_keep.append(item)
        tagged_tokens = tagged_tokens_keep

    return tagged_tokens


def extract_tags(tagged_tokens: List[Tuple[str, str]]) -> str:
    """WIP."""
    tags = []

    for pair in tagged_tokens:

        _, tag = pair
        tags.append(tag)

    tags = "_".join(tags)

    return tags


def apply_tag_rule(rule: str, df: DataFrame, vip_tag: str, feat_name: str, summary: bool = True):
    """Apply an operation on the dataframe based on tags.

    Parameters
    ----------
    rule : str
        Rule to use, either "count" for the number of found instances, "set" to transform the
        result into a set or "avg" to compute the average length of found instances.

    df : DataFrame
        A dataframe containing the corresponding column, either "tags" if **summary** is True,
        or "d_tags" if **summary** is False.

    vip_tag : str
        Tag to apply the **rule** to.

    feat_name : str
        Name of the new feature to append to the dataframe.

    summary : bool, default=True
        If True, will apply the rule on the "tags" column of the dataframe, otherwise on the
        "d_tags" column.

    Raises
    ------
    ValueError
        If the corresponding column "summary" or "document" is not present in **df**, depending on
        the value of **summary**. If the **rule** is not supported.
    """
    if feat_name not in df.columns:

        # Choose the column to which the rule must be applied
        col = "summary_tagged" if summary else "document_tagged"

        if col not in df.columns:

            err_msg = f"Column {col} not present in the dataframe."
            raise ValueError(err_msg)

        # Apply the rule and put the result in a new column
        if rule == "count":

            def count_tag(tagged_tokens: List[Tuple[str, str]], vip_tag: str) -> float:

                cnt = 0.
                for tagged_token in tagged_tokens:
                    if tagged_token[1] == vip_tag:
                        cnt += 1

                return cnt

            df[feat_name] = df[col].apply(lambda x: count_tag(x, vip_tag))

        elif rule == "set":

            def set_tag(tagged_tokens: List[Tuple[str, str]], vip_tag: str) -> Set[str]:

                ens = set()
                for tagged_token in tagged_tokens:
                    if tagged_token[1] == vip_tag:
                        ens.add(tagged_token[0])

                return ens

            df[feat_name] = df[col].apply(lambda x: set_tag(x, vip_tag))

        elif rule == "avg":

            def avg_tag(tagged_tokens: List[Tuple[str, str]], vip_tag: str) -> float:

                avg = 0.
                n = 0
                for tagged_token in tagged_tokens:
                    if tagged_token[1] == vip_tag:
                        avg += len(tagged_token[0])
                        n += 1
                avg = avg / n if n > 0 else 0.

                return avg

            df[feat_name] = df[col].apply(lambda x: avg_tag(x, vip_tag))

        else:

            err_msg = f"Unknown tagging rule {rule}."
            raise ValueError(err_msg)


def create_tagging_feat(
    seed: int, train_df: DataFrame, test_df: Optional[DataFrame] = None,
    tag_feat: Optional[List[str]] = None, more_docs: Optional[DataFrame] = None,
    delta: float = 0.1, b: float = 0.5, k_1: float = 1., svd_components: int = 1
):
    """Auxiliary function to create tagging features.

    Parameters
    ----------
    seed : int
        Seed to use everywhere for reproducibility.

    train_df : DataFrame
        Training dataframe containing "summary" and "document" columns.

    test_df : optional DataFrame, default=None
        Test dataframe containing "summary" and "document" columns.

    tag_feat : optional list of str, default=None
        List of all tagging features to compute. The name of a tagging feature must be of the form
        "A_B". A is a tag from all available tags in `nltk` or "tags". B is the type of feature we
        want, it can be "count" for the number of instances, "avg" for the average length of
        instances, "overlap" for the number of instances both found in the summary and the original
        document, "ratio" for the ratio between the number of instances found in the summary and
        the number of instances found in the document or "C_D" where C and D are parameters for idf
        features.

    more_docs : optional DataFrame, default=None
        Additional dataframe containing a "document" column to use for idf computation.

    delta : float, default=0.1
        Parameter value for "d" composition in tf-idf.

    b : float, default=0.5
        Parameter value for "p" composition in tf-idf.

    k_1 : float, default=1.0
        Parameter value for "k" composition in tf-idf.

    svd_components : int, default=1
        Number of components for the truncated SVD component analysis. This number must be smaller
        than the number of features in the counter.

    Raises
    ------
    ValueError
        If one of the features is not supported. If one of the tags is not in the available
        nltk tags.
    """
    if tag_feat is None or len(tag_feat) == 0:
        return

    print("\nComputing tagging features...")

    init_feat = set(train_df.columns)
    new_feat = set()

    dfs = [train_df]
    if test_df is not None:
        dfs.append(test_df)

    stopwords = stopwords_collection.words("english")
    punct = string.punctuation.replace("-", "")
    vip_tag = ""

    for feat_name in tag_feat:

        # Extract feature options
        feat_split = feat_name.split("_")
        vip_tag = feat_split[0]
        feat_type = "_".join(feat_split[1:])

        if vip_tag not in ALL_NLTK_TAGS + ["tags"]:

            err_msg = f"Unsupported nltk tag {vip_tag}."
            raise ValueError(err_msg)

        # Compute tags of summaries and eventually of documents
        for df in dfs:

            if "summary_tagged" not in df.columns:

                df["summary_tagged"] = df["summary"].apply(
                    lambda x: text_to_tagged_tokens(
                        x, stopwords, punct, remove_stopwords=False, stemming=False,
                        pos_filtering=False,
                    )
                )

            if (
                (feat_type in {"ratio", "overlap"} or vip_tag == "tags") and
                "document_tagged" not in df.columns
            ):

                df["document_tagged"] = df["document"].apply(
                    lambda x: text_to_tagged_tokens(
                        x, stopwords, punct, remove_stopwords=False, stemming=False,
                        pos_filtering=False
                    )
                )

        if (
            vip_tag == "tags" and
            (more_docs is not None and "document_tagged" not in more_docs.columns)
        ):

            more_docs["document_tagged"] = more_docs["document"].apply(
                lambda x: text_to_tagged_tokens(
                    x, stopwords, punct, remove_stopwords=False, stemming=False,
                    pos_filtering=False
                )
            )

        # Compute tagging feature
        if vip_tag == "tags":

            suffix = "_" + vip_tag
            max_features = 100
            svd_components = 25

            train_df["summary_tags"] = train_df["summary_tagged"].apply(extract_tags)
            train_df["document_tags"] = train_df["document_tagged"].apply(extract_tags)
            if test_df is not None:
                test_df["summary_tags"] = test_df["summary_tagged"].apply(extract_tags)
                test_df["document_tags"] = test_df["document_tagged"].apply(extract_tags)
            if more_docs is not None:
                more_docs["document_tags"] = more_docs["document_tagged"].apply(extract_tags)

            create_idf_feat(
                seed, train_df, test_df=test_df, more_docs=more_docs, idf_feat=[feat_type],
                max_features=max_features, delta=delta, b=b, k_1=k_1,
                svd_components=svd_components, suffix=suffix
            )

            if feat_type == "count":
                new_cols = [feat_name + suffix + f"_PCA_{i}" for i in range(1, svd_components + 1)]
            elif feat_type == "lda":
                new_cols = [feat_name + suffix + f"_{i}" for i in range(3)]
            elif feat_type == "idf":
                new_cols = [feat_name + suffix + f"_PCA_{i}" for i in range(1, svd_components + 1)]
            elif feat_type == "cos":
                new_cols = [feat_name + suffix]
            else:
                new_cols = []

            new_feat.union(set(new_cols))

        else:

            for df in dfs:

                if feat_type == "count":

                    apply_tag_rule("count", df, vip_tag, feat_name)

                elif feat_type == "avg":

                    apply_tag_rule("avg", df, vip_tag, feat_name)

                elif feat_type == "overlap":

                    s_feat_name = vip_tag + "_set"
                    d_feat_name = "d_" + vip_tag + "_set"

                    apply_tag_rule("set", df, vip_tag, s_feat_name)
                    apply_tag_rule("set", df, vip_tag, d_feat_name, summary=False)
                    df[feat_name] = [
                        len(x[0] & x[1]) for x in df[[s_feat_name, d_feat_name]].values
                    ]

                elif feat_type == "ratio":

                    s_feat_name = vip_tag + "_count"
                    d_feat_name = "d_" + vip_tag + "_count"

                    apply_tag_rule("count", df, vip_tag, s_feat_name)
                    apply_tag_rule("count", df, vip_tag, d_feat_name, summary=False)
                    df[feat_name] = df[s_feat_name] / df[d_feat_name]

                else:

                    err_msg = f"From tagging feature {feat_name},"
                    err_msg += f" unsupported feature type {feat_type}."
                    raise ValueError(err_msg)

            new_feat.add(feat_name)

    # Drop intermediary features
    inter_feat = set(train_df.columns).difference(init_feat.union(new_feat))

    train_df.drop(columns=inter_feat, inplace=True)
    if test_df is not None:
        test_df.drop(columns=inter_feat, inplace=True)

    print(f"Number of tagging features: {len(new_feat)}.")
