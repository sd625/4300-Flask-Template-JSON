import numpy as np

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

#list of programs and the reviews associated with them
programs = []

#to get from user
query = ""

def review_svd(review_df, k_value = 100):
  
  #information from reviews
  review_tokens = []
  for i in review_df["review"]:
    review_tokens.append(i.lower())

  #vectorizer for review tokens
  vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.9, min_df = 0.01)
  review_td_matrix = vectorizer.fit_transform(review_tokens)


  docs_compressed, s, words_compressed = svds(review_td_matrix, k=40)
  words_compressed = words_compressed.transpose()

  docs_compressed_normed = normalize(docs_compressed)
  
  top_programs = closest_programs_to_query(query, vectorizer, docs_compressed_normed, words_compressed)

  return top_programs

def closest_programs_to_query(query, vectorizer, docs_compressed_normed, words_compressed, k = 5):
    #transform query to appropriate shape
    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()

    sims = docs_compressed_normed.dot(query_vec)
    asort = np.argsort(-sims)[:k+1]
    return [(i, programs[i][0],sims[i]) for i in asort[1:]]