import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
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
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_vectorized, y_train)

# glance at rf accuracy
rf_pred = rf_classifier.predict(X_test_vectorized)
rf_report = classification_report(y_test, rf_pred)


# program and median rating
group = data.groupby("program")["review_rating"].median()
program_rating = {f"{name}": rating for i, (name, rating) in enumerate(group.items())}


def sentiment_ranking(
    query,
    filtered_programs,  # from edit distance
    vectorizer=vectorizer,
    classifier=rf_classifier,
):
    q_vec = vectorizer.transform([query])
    q_rating = classifier.predict(q_vec)[0]
    rating_diff = []
    for program in filtered_programs:
        name = program["program"]
        if name in program_rating:
            r = program_rating[name]
        else:
            r = 0
        diff = abs(q_rating - r)
        rating_diff.append(diff)
    # ranking
    pair = list(zip(filtered_programs, rating_diff))
    sorted_pairs = sorted(pair, key=lambda x: x[1])

    # return programs listed in ranked order
    ranked = [pair[0] for pair in sorted_pairs]
    print(ranked[0])
    # create json data
    json_data = ranked.copy()
    for program in json_data:
        name = program["program"]
        if name in program_rating:
            r = program_rating[name]
        else:
            r = 0
        program["rating"] = r

    return json_data
