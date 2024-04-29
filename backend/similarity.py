import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics.distance import edit_distance
from numpy import linalg as LA

# combine edit distance with cosine similarity


def compute_cosine_similarity(query, tokens):

    # Convert texts to vectors
    text = " ".join(list(tokens))
    vectorizer = CountVectorizer().fit([query, text])
    q_vec, tokens_vec = vectorizer.transform([query, text])

    # Compute cosine similarity -- use sklearn because can handle different sized arrays
    cosine_sim = cosine_similarity(q_vec, tokens_vec)[0][0]
    return cosine_sim


def combine_similarity_scores(
    cosine_sim, normalized_edit_dist, cosine_weight=0.5, edit_dist_weight=0.5
):
    # Combine cosine similarity and normalized edit distance scores
    combined_score = (
        cosine_weight * cosine_sim + edit_dist_weight * normalized_edit_dist
    )
    return combined_score


def compute_composite_similarity(
    query, review_tokens, normalized_edit_dist, cosine_weight=0.8, edit_dist_weight=0.2
):
    cosine_sim = compute_cosine_similarity(query, review_tokens)
    composite_similarity = combine_similarity_scores(
        cosine_sim, normalized_edit_dist, cosine_weight, edit_dist_weight
    )
    return composite_similarity
