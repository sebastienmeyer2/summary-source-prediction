"""Statistical detection of human-written or GPT-2/BERT generated texts.

Code adapted from https://github.com/HendrikStrobelt/detecting-fake-text.

References
----------
.. [1] Sebastian Gehrmann, Hendrik Strobelt and Alexander M. Rush. *GLTR: Statistical Detection and
    Visualization of Generated Text.* June 2019. (Available at: https://arxiv.org/abs/1906.04043)
"""


from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import torch
from torch import Tensor

from transformers import GPT2LMHeadModel, GPT2Tokenizer


tqdm.pandas()


class AbstractLanguageChecker:
    """Abstract Class that defines the Backend API of GLTR.

    To extend the GLTR interface, you need to inherit this and fill in the defined functions.
    """

    def __init__(self):
        """Load all necessary components for the other functions.

        Typically, this will comprise a tokenizer and a model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def check_probabilities(self, in_text: str, topk: int = 40) -> Dict[str, Any]:
        """Function that GLTR interacts with to check the probabilities of words.

        Parameters
        ----------
        in_text : str
            The text that you want to check.

        topk : int, default=40
            Your desired truncation of the head of the distribution.

        Returns
        -------
        payload : dict
            The wrapper for results in this function, described below.

        Notes
        -----
            Values in the **payload** returned dict:
            - bpe_strings: list of str -- Each individual token in the text
            - real_topk: list of tuples -- (ranking, prob) of each token
            - pred_topk: list of list of tuple -- (word, prob) for all topk
        """
        raise NotImplementedError

    def probabilities_feat(self, df: Series, nb_bins: Dict[str, int], topk: int = 40) -> Series:
        """Function that GLTR interacts with to add topk features to a data set.

        Parameters
        ----------
        df : Series
            Initial data set with a "summary" column in it.

        bins : dict of {str: int}
            Dictionary which associates a feature to its number of bins.

        topk : int, default=40
            Your desired truncation of the head of the distribution.

        Returns
        -------
        df : Series
            Transformed data set with appended topk features.
        """
        raise NotImplementedError()

    def postprocess(self, token: str) -> str:
        """Clean up the tokens from any special chars and encode.

        The leading space is encoded by UTF-8 code "\u0120", linebreak with UTF-8 code 266
        "\u010A".

        Parameters
        ----------
        token : str
            Raw token text.

        Returns
        -------
        str
            Cleaned and re-encoded token text.
        """
        raise NotImplementedError


class LM(AbstractLanguageChecker):
    """See `AbstractLanguageChecker` class for description."""

    def __init__(self, model_name_or_path: str = "gpt2"):
        """Initialize a GPT-2 language checker.

        Parameters
        ----------
        model_name_or_path : str, default="gpt2"
            Name of the model or path to the pretrained file.
        """
        super().__init__()

        self.enc = GPT2Tokenizer.from_pretrained(
            model_name_or_path, cache_dir="data/transformers_cache"
        )

        self.model = GPT2LMHeadModel.from_pretrained(
            model_name_or_path, cache_dir="data/transformers_cache"
        )
        self.model.to(self.device)
        self.model.eval()

        self.start_token = self.enc(self.enc.bos_token, return_tensors="pt").data["input_ids"][0]

        print("Loaded GPT-2 model!")

    def check_probabilities(self, in_text: str, topk: int = 40) -> Dict[str, Any]:
        """See `AbstractLanguageChecker` class for description."""
        # Process input
        token_ids = self.enc(in_text, return_tensors="pt").data["input_ids"][0]
        token_ids = torch.concat([self.start_token, token_ids])

        # Forward through the model
        output = self.model(token_ids.to(self.device))
        all_logits = output.logits[:-1].detach().squeeze()

        # Construct target and pred
        # yhat = torch.softmax(logits[0, :-1], dim=-1)
        all_probs = torch.softmax(all_logits, dim=1)
        y = token_ids[1:]

        # Sort the predictions for each timestep
        sorted_preds = torch.argsort(all_probs, dim=1, descending=True).cpu()

        # Format [(pos, prob), ...]
        real_topk_pos = [
            int(np.where(sorted_preds[i] == y[i].item())[0][0]) for i in range(y.shape[0])
        ]
        real_topk_probs = all_probs[np.arange(0, y.shape[0], 1), y].data.cpu().numpy().tolist()
        real_topk_probs = list(map(lambda x: round(x, 5), real_topk_probs))

        real_topk = list(zip(real_topk_pos, real_topk_probs))

        # Format [str, str, ...]
        bpe_strings = self.enc.convert_ids_to_tokens(token_ids[:])
        bpe_strings = [self.postprocess(s) for s in bpe_strings]  # pylint: disable=not-an-iterable

        topk_prob_values, topk_prob_inds = torch.topk(all_probs, k=topk, dim=1)

        pred_topk = [
            list(zip(
                self.enc.convert_ids_to_tokens(topk_prob_inds[i]),
                topk_prob_values[i].data.cpu().numpy().tolist()
            )) for i in range(y.shape[0])
        ]
        pred_topk = [[(self.postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk]

        payload = {
            "bpe_strings": bpe_strings,
            "real_topk": real_topk,
            "pred_topk": pred_topk
        }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return payload

    def probabilities_feat(self, df: pd.Series, nb_bins: Dict[str, int], topk: int = 40) -> Series:
        """See `AbstractLanguageChecker` class for description."""
        # Process input
        in_text = df["summary"]

        # Compute the features
        payload = self.check_probabilities(in_text, topk=topk)
        real_topk = payload["real_topk"]

        eps = 1e-7

        if "count" in nb_bins:

            topk_count = np.array([real_topk[i][0] for i in range(len(real_topk))])
            count_bins = np.linspace(
                np.log(min(topk_count) + eps) - eps, np.log(max(topk_count) + eps) + eps,
                num=nb_bins["count"] + 1
            )
            count_hist, _ = np.histogram(topk_count, bins=count_bins)

            count_feat_names = [f"topk_count_{i}" for i in range(len(count_bins) - 1)]

            for count_name, count_feat in zip(count_feat_names, count_hist):
                df[count_name] = count_feat

        if "frac" in nb_bins:

            topk_frac = np.array([real_topk[i][1] for i in range(len(real_topk))])
            frac_bins = np.linspace(
                min(topk_frac) - eps, max(topk_frac) + eps, num=nb_bins["frac"] + 1
            )
            frac_hist, _ = np.histogram(topk_frac, bins=frac_bins)

            frac_feat_names = [f"topk_frac_{i}" for i in range(len(frac_bins) - 1)]

            for frac_name, frac_feat in zip(frac_feat_names, frac_hist):
                df[frac_name] = frac_feat

        return df

    def sample_unconditional(
        self, length: int = 100, topk: int = 5, temperature: float = 1.
    ) -> str:
        """Sample length words from the model.

        Parameters
        ----------
        length : int, default=100
            Number of words to sample.

        topk : int, default=5
            Truncation of the head of the distribution.

        temperature : float, default=1.0
            Temperature for rescaling the predicted logits.

        Returns
        -------
        output_text : str
            Generated words.
        """
        context = torch.full(
            (1, 1), self.enc.encoder[self.start_token], device=self.device, dtype=torch.long
        )
        prev = context
        output = context
        past = None

        # Forward through the model
        with torch.no_grad():
            for _ in range(length):

                logits, past = self.model(prev, past=past)
                logits = logits[:, -1, :] / temperature

                # Filter predictions to topk and softmax
                probs = torch.softmax(top_k_logits(logits, k=topk), dim=-1)

                # Sample
                prev = torch.multinomial(probs, num_samples=1)

                # Construct output
                output = torch.cat((output, prev), dim=1)

        output_text = self.enc.decode(output[0].tolist())

        return output_text

    def postprocess(self, token: str) -> str:
        """See `AbstractLanguageChecker` class for description."""
        with_space = False
        with_break = False

        if token.startswith("Ġ"):
            with_space = True
            token = token[1:]

        elif token.startswith("â"):
            token = " "

        elif token.startswith("Ċ"):
            token = " "
            with_break = True

        token = "-" if token.startswith("â") else token
        token = '“' if token.startswith("ľ") else token
        token = '”' if token.startswith("Ŀ") else token
        token = "'" if token.startswith("Ļ") else token

        if with_space:
            token = "\u0120" + token
        if with_break:
            token = "\u010A" + token

        return token


def top_k_logits(logits: Tensor, k: int) -> Tensor:
    """Filter logits to only the top k choices.

    Parameters
    ----------
    logits : Tensor
        Predicted logits.

    Returns
    -------
    topk : Tensor
        Filtered top-k predicted logits with near zero elsewhere.
    """
    if k == 0:
        return logits

    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]

    topk = torch.where(
        logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits
    )

    return topk


def create_gltr_feat(
    train_df: DataFrame, test_df: Optional[DataFrame] = None,
    gltr_feat: Optional[List[str]] = None
) -> Tuple[DataFrame, Optional[DataFrame]]:
    """Auxiliary function to create GLTR features.

    Parameters
    ----------
    train_df : DataFrame
        Training dataframe containing a "summary" column.

    test_df : optional DataFrame, default=None
        Test dataframe containing a "summary" column.

    gltr_feat : optional list of str, default=None
        List of all GLTR features to compute. The name of a GLTR feature should be of the form
        "A_B". A is the name of a topk value computed by GLTR, it can be "count" or "frac". B is
        the number of bins to compute the feature.

    Raises
    ------
    ValueError
        If one of the features is not supported.
    """
    if gltr_feat is None or len(gltr_feat) == 0:
        return train_df, test_df

    print("\nComputing GLTR features...")

    init_feat = set(train_df.columns)

    dfs = [train_df]
    if test_df is not None:
        dfs.append(test_df)

    # Features parameters
    gltr_bins = {}

    for feat_name in gltr_feat:

        feat_split = feat_name.split("_")
        feat_type = feat_split[0]
        feat_bins = int(feat_split[1])

        if feat_type not in {"count", "frac"}:

            err_msg = f"From GLTR feature {feat_name}, unsupported feature type {feat_type}."
            raise ValueError(err_msg)

        gltr_bins[feat_type] = feat_bins

    # Compute features
    lm = LM()

    for j, df in enumerate(dfs):

        dfs[j] = df.progress_apply(
            lambda x: lm.probabilities_feat(x, gltr_bins, topk=5),
            axis=1
        )

    train_df = dfs[0]
    if test_df is not None:
        test_df = dfs[1]

    print(f"Number of GLTR features: {train_df.shape[1] - len(init_feat)}.")

    return train_df, test_df
