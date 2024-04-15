import numpy as np

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

import json

# import matplotlib
# import matplotlib.pyplot as plt


def review_svd(programs_df, query, k_value = 100):
  documents = []
  # print(programs_df.head())
  # print(programs_df['tokens'])
  for index, row in programs_df.iterrows():

    program_id = row['id']
    program_name = row['program']
    program_location = row['location']
    program_url = row['url']
    
    program_reviews = row['reviews']
    program_tokens =  " ".join(row['tokens'])
    #print(program_tokens)
    doc = (program_id, program_name, program_location, program_url, program_reviews, program_tokens)
    documents.append(doc)

  #print("Documents", documents)

  
  vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.9, min_df = 0.01)
  review_td_matrix = vectorizer.fit_transform([x[5] for x in documents])

  word_to_index = vectorizer.vocabulary_
  #print(word_to_index)
  index_to_word = {i:t for t,i in word_to_index.items()}


  docs_compressed, s, words_compressed = svds(review_td_matrix, k=40)
  words_compressed = words_compressed.transpose()
  

  # plt.plot(s[::-1])
  # plt.xlabel("Singular value number")
  # plt.ylabel("Singular value")
  # plt.show()

  # for i in range(40):
  #   print("Top words in dimension", i)
  #   dimension_col = words_compressed[:,i].squeeze()
  #   asort = np.argsort(-dimension_col)
  #   print([index_to_word[i] for i in asort[:10]])
  #   print()

  word = 'spanish'
  words_compressed_normed = normalize(words_compressed, axis = 1)
  # print("Using SVD:")
  # for i in closest_words(word_to_index, index_to_word, word, words_compressed_normed):
  #   try:
  #     #print("{}, {:.3f}".format(w, sim))
  #     print(i)
  #   except:
  #     print("word not found")
      
  # print()

  docs_compressed_normed = normalize(docs_compressed)
  
  top_programs = closest_programs_to_query(query, documents, vectorizer, docs_compressed_normed, words_compressed)

  # return as a json string (can just copy code from lines 64-68 with id, sim, program, program loc)
  return top_programs

def closest_words(word_to_index, index_to_word, word_in, words_representation_in, k = 10):
    if word_in not in word_to_index: return "Not in vocab."
    sims = words_representation_in.dot(words_representation_in[word_to_index[word_in],:])
    asort = np.argsort(-sims)[:k+1]
    return [(index_to_word[i],sims[i]) for i in asort[1:]]

def closest_programs_to_query(query, documents, vectorizer, docs_compressed_normed, words_compressed, k = 5):
    #transform query to appropriate shape
    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()

    sims = docs_compressed_normed.dot(query_vec)
    asort = np.argsort(-sims)[:k+1]
    # include program location in output as well (see line 58 in app.py)

    # for i in asort[1:]:
    #    print(sims[i])

    

    json_data = [
    {"id": documents[i][0], "program": documents[i][1], "program_location": documents[i][2],
      "program_url": documents[i][3], "program_reviews": documents[i][4], "program_tokens": documents[i][5]}
    for i in asort[1:]]

    json_string = json.dumps(json_data, indent=2)

    return json_string
