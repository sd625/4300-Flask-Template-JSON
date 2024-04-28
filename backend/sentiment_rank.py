import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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

# vectorizer
count_vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# naive bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)

# glance at nb accuracy
nb_pred = nb_classifier.predict(X_test_vectorized)
nb_report = classification_report(y_test, nb_pred)

# random forest classifier is better at accounting for non linear relationships
rf_classifer = RandomForestClassifier()
rf_classifer.fit(X_train_vectorized, y_train)

# glance at rf accuracy
rf_pred = rf_classifer.predict(X_test_vectorized)
rf_report = classification_report(y_test, rf_pred)


# program and median rating
group = data.groupby("program")["review_rating"].median()
program_rating = {
    f"{name}": rating for i, (name, rating) in enumerate(group.items())
}


def sentiment_ranking(
    query,
    filtered_programs,  # programs from edit distance
    vectorizer=vectorizer,
    classifier=rf_classifier,
):
    if query == "": return filtered_programs
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

    # create json data
    json_data = [
        {
            "id": id,
            "program": program_name,
            "program_location": program_location,
            "url": program_url,
            "rating" : program_rating[program_name]
        }
        for id, _, program_name, program_location, program_url in ranked
    ]

    return json_data
