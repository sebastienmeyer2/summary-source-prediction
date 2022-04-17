"""Compute features based on regex expressions."""


from typing import List, Optional

from pandas import DataFrame


def apply_regex_rule(rule: str, df: DataFrame, expr: str, feat_name: str, summary: bool = True):
    """Apply a regex expression and perform an operation on the results of the query.

    Parameters
    ----------
    rule : str
        Rule to use after applying the regex **expr**, either "count" for the number of found
        instances, "set" to transform the result into a set or "avg" to compute the average length
        of found instances.

    df : DataFrame
        A dataframe containing the corresponding column, either "summary" if **summary** is True,
        or "document" if **summary** is False.

    expr : str
        The regex expression to apply.

    feat_name : str
        Name of the new feature to append to the dataframe.

    summary : bool, default=True
        If True, will apply the regex **expr** on the "summary" column of the dataframe, otherwise
        on the "document" column.

    Raises
    ------
    ValueError
        If the corresponding column "summary" or "document" is not present in **df**, depending on
        the value of **summary**. If the **rule** is not supported.
    """
    if feat_name not in df.columns:

        # Choose the column to which the rule must be applied
        col = "summary" if summary else "document"

        if col not in df.columns:

            err_msg = f"Column {col} not present in the dataframe."
            raise ValueError(err_msg)

        # Apply the rule and put the result in a new column
        if rule == "count":

            df[feat_name] = df[col].str.findall(expr).str.len().astype(float)

        elif rule == "set":

            df[feat_name] = df[col].str.findall(expr).apply(set)

        elif rule == "avg":

            def avg(arr_of_str):
                n = len(arr_of_str)
                mean = 0.
                if n == 0:
                    return mean
                for x in arr_of_str:
                    mean += len(x)
                mean /= n
                return mean

            col = "summary" if summary else "document"
            df[feat_name] = df[col].str.findall(expr).apply(avg)

        else:

            err_msg = f"Unknown regex rule {rule}."
            raise ValueError(err_msg)


def regex_name_to_expr(feat_regex_name: str) -> str:
    """Convert a feature name to its corresponding regex expression.

    Parameters
    ----------
    feat_regex_name : str
        Name of the feature to compute.

    Returns
    -------
    expr : str
        Corresponding regex expression.

    Raises
    ------
    ValueError
        If the **feat_regex_name** is not supported.
    """
    # Features associated to length
    if feat_regex_name == "char":

        # This expression captures all characters, even commas, periods, dashes, numbers and white
        # spaces.
        expr = r"."

    elif feat_regex_name == "word":

        # This expression captures all words, which are separated either by commas, periods or
        # dashs. Please note that this expression might capture numbers aswell.
        expr = r"\b\w+\b"

    elif feat_regex_name == "sent":

        # This expression captures all sentences
        expr = r"([A-Z][^\.!?]*[\.!?])"

    # Features associated to uppercase, lowercase and special characters
    elif feat_regex_name == "upper_char":

        # This expression captures all upper characters
        expr = r"[A-Z]"

    elif feat_regex_name == "upper_word":

        # This expression captures all words starting with a capital letter
        expr = r"\b[A-Z]\w+"

    elif feat_regex_name == "numeric":

        # This expression captures all numbers which might be separated either by commas, periods
        # or dashs
        expr = r"[0-9]"

    elif feat_regex_name == "dash":

        # This expression captures all dashes
        expr = r"[-]"

    # Features associated to groups
    elif feat_regex_name == "group":

        # This expression captures all groups of consecutive words which start with a capital
        # letter
        expr = r"([A-Z][\w-]*(?:\s+[A-Z][\w-]*)+)"

    # Features associated to beginning and end of string
    elif feat_regex_name == "is_first_upper":

        # This expression captures the first character iff it is uppercase
        expr = r"^[A-Z]"

    elif feat_regex_name == "is_last_ponct":

        # This expression captures the last character iff it is a ponctuation
        expr = r"[?.!,:;']$"

    elif feat_regex_name == "space_before_ponct":

        # This expression captures all white spaces which are before ponctuation
        expr = r"[ ][?.!,:;']"

    elif "word_" in feat_regex_name:

        n = feat_regex_name.split("_")[-1]

        if n not in [f"{i}" for i in range(1, 11)]:

            err_msg = f"Unsupported number of chars in word {n}."
            raise ValueError(err_msg)

        # This expression captures all words of specific length which are separated either by
        # commas, periods or dashs. Please note that it might capture numbers aswell.
        expr = fr"\b\w{{{n}}}\b"

    else:

        err_msg = f"Unsupported regex feature {feat_regex_name}."
        raise ValueError(err_msg)

    return expr


def create_regex_feat(
    train_df: DataFrame, test_df: Optional[DataFrame] = None,
    regex_feat: Optional[List[str]] = None
):
    """Auxiliary function to create regex features.

    Parameters
    ----------
    train_df : DataFrame
        Training dataframe containing "summary" and "document" columns.

    test_df : optional DataFrame, default=None
        Test dataframe containing "summary" and "document" columns.

    regex_feat : optional list of str, default=None
        List of all regex features to compute. The name of a regex feature should be of the form
        "A_B". A is the name of the regex expression, for example "upper_word". B is the type of
        feature we want, it can be "count" for the number of instances, "avg" for the average
        length of instances, "overlap" for the number of instances both found in the summary and
        the original document or "ratio" for the ratio between the number of instances found in the
        summary and the number of instances found in the document.

    Raises
    ------
    ValueError
        If one of the features is not supported.
    """
    if regex_feat is None or len(regex_feat) == 0:
        return

    print("\nComputing regex features...")

    init_feat = set(train_df.columns)
    new_feat = set()

    dfs = [train_df]
    if test_df is not None:
        dfs.append(test_df)

    for feat_name in regex_feat:

        for df in dfs:

            feat_split = feat_name.split("_")
            feat_type = feat_split[-1]
            feat_regex_name = "_".join(feat_split[:-1])
            feat_regex_expr = regex_name_to_expr(feat_regex_name)

            if feat_type == "count":

                apply_regex_rule("count", df, feat_regex_expr, feat_name)

            elif feat_type == "avg":

                apply_regex_rule("avg", df, feat_regex_expr, feat_name)

            elif feat_type == "overlap":

                s_feat_name = feat_regex_name + "_set"
                d_feat_name = "d_" + feat_regex_name + "_set"

                apply_regex_rule("set", df, feat_regex_expr, s_feat_name)
                apply_regex_rule("set", df, feat_regex_expr, d_feat_name, summary=False)
                df[feat_name] = [
                    len(x[0] & x[1]) for x in df[[s_feat_name, d_feat_name]].values
                ]

            elif feat_type == "ratio":

                s_feat_name = feat_regex_name + "_count"
                d_feat_name = "d_" + feat_regex_name + "_count"

                apply_regex_rule("count", df, feat_regex_expr, s_feat_name)
                apply_regex_rule("count", df, feat_regex_expr, d_feat_name, summary=False)
                df[feat_name] = df[s_feat_name] / df[d_feat_name]

            else:

                err_msg = f"From regex feature {feat_name}, unsupported feature type {feat_type}."
                raise ValueError(err_msg)

        new_feat.add(feat_name)

    # Drop intermediary features
    inter_feat = set(train_df.columns).difference(init_feat.union(new_feat))

    train_df.drop(columns=inter_feat, inplace=True)
    if test_df is not None:
        test_df.drop(columns=inter_feat, inplace=True)

    print(f"Number of regex features: {len(new_feat)}.")
