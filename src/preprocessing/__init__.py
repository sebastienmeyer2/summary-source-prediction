"""Functions to preprocess the data by reading, computing and/or removing features."""


from preprocessing.reduction.droping import drop_corr_feat, drop_list_feat
from preprocessing.reduction.pca import perform_pca
from preprocessing.features.embeddings import create_embed_feat
from preprocessing.features.gltr import create_gltr_feat
from preprocessing.features.polynomial import create_poly_feat
from preprocessing.features.manual_regex import create_regex_feat
from preprocessing.features.pos_tagging import create_tagging_feat
from preprocessing.features.tfidf import create_idf_feat


__all__ = [
    "drop_corr_feat",
    "drop_list_feat",
    "perform_pca",
    "create_embed_feat",
    "create_gltr_feat",
    "create_poly_feat",
    "create_regex_feat",
    "create_tagging_feat",
    "create_idf_feat"
]
