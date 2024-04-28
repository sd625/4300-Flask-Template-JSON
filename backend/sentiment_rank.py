import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import json
from sklearn.metrics import classification_report


data = pd.read_csv("all-study-abroad-reviews.csv")

reviews_data = data["review"]
ratings = np.round(data["review_rating"])

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    reviews_data, ratings, test_size=0.15, random_state=42
)

# count vectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# naive bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)

# glance at nb accuracy
nb_pred = nb_classifier.predict(X_test_vectorized)
nb_report = classification_report(y_test, nb_pred)

# program and median rating
group = data.groupby("program")["review_rating"].median()
program_rating = {
    f"{name}_{i}": rating for i, (name, rating) in enumerate(group.items())
}


def sentiment_ranking(
    query,
    filtered_programs,  # programs from ranked_edit_distance
    vectorizer=vectorizer,
    classifier=nb_classifier,
):
    q_vec = vectorizer.transform([query])
    q_rating = classifier.predict(q_vec)[0]
    rating_diff = []
    for program in filtered_programs:
        name = program["program"]
        diff = abs(q_rating - program_rating[name])
        rating_diff.append(diff)
    # ranking
    pair = list(zip(filtered_programs, rating_diff))
    sorted_pairs = sorted(pair, key=lambda x: x[0])

    # return programs listed in ranked order
    ranked = [pair[1] for pair in sorted_pairs]

    # create json string
    json_data = [
        {
            "id": id,
            "program": program_name,
            "program_location": program_location,
            # "program_url": url,
        }
        for id, _, program_name, program_location in ranked
    ]

    json_string = json.dumps(json_data, indent=2)

    return json_data
