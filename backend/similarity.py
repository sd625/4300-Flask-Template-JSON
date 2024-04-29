import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics.distance import edit_distance


def compute_cosine_similarity(query, review_tokens):
    # Convert texts to vectors
    vectorizer = CountVectorizer().fit([query, review_tokens])
    vector_query, vector_review_tokens = vectorizer.transform([query, review_tokens])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(vector_query, vector_review_tokens)[0][0]
    return cosine_sim


# def compute_normalized_edit_distance(query, review_tokens):
#     # Compute edit distance
#     edit_dist = edit_distance(query, review_tokens)

#     # Normalize edit distance
#     max_length = max(len(query), len(review_tokens))
#     normalized_edit_dist = edit_dist / max_length

#     return 1 - normalized_edit_dist  # Convert to similarity score


def combine_similarity_scores(
    cosine_sim, normalized_edit_dist, cosine_weight=0.7, edit_dist_weight=0.3
):
    # Combine cosine similarity and normalized edit distance scores
    combined_score = (
        cosine_weight * cosine_sim + edit_dist_weight * normalized_edit_dist
    )
    return combined_score


def compute_composite_similarity(
    query, review_tokens, normalized_edit_dist, cosine_weight=0.7, edit_dist_weight=0.3
):
    cosine_sim = compute_cosine_similarity(query, " ".join(review_tokens))
    print("cosine sim", cosine_sim)
    composite_similarity = combine_similarity_scores(
        cosine_sim, normalized_edit_dist, cosine_weight, edit_dist_weight
    )
    return composite_similarity
