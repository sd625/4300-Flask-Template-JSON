import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


data = pd.read_csv("all-study-abroad-reviews.csv")

reviews_data = data["review"]
ratings = np.round(data["review_rating"])

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    reviews_data, ratings, test_size=0.2, random_state=42
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

# glance at rf accuracy (~67%)
rf_pred = rf_classifier.predict(X_test_vectorized)
rf_report = classification_report(y_test, rf_pred)


# program and median rating
group = data.groupby("program")["review_rating"].median()
program_rating = {f"{name}": rating for i, (name, rating) in enumerate(group.items())}


# get program rating return 0 if program has no reviews
def rating(name):
    if name in program_rating:
        return program_rating[name]
    else:
        return 0


# create json data
def create_json_data(program_dict):
    json_data = program_dict.copy()
    for program in json_data:
        program_name = program["program"]
        program["rating"] = rating(program_name)
    return json_data


def sentiment_ranking(
    query,
    filtered_programs,  # from edit distance
    vectorizer=vectorizer,
    classifier=rf_classifier,
):

    if query == "":
        return create_json_data(filtered_programs)

    q_vec = vectorizer.transform([query])
    q_rating = classifier.predict(q_vec)[0]
    rating_diff = []
    for program in filtered_programs:
        name = program["program"]
        diff = abs(q_rating - rating(name))
        rating_diff.append(diff)
    # ranking
    pair = list(zip(filtered_programs, rating_diff))
    sorted_pairs = sorted(pair, key=lambda x: x[1])

    # return programs listed in ranked order
    ranked = [pair[0] for pair in sorted_pairs]
    # print(ranked[0])
    # create json data
    json_data = ranked.copy()
    for program in json_data:
        program["rating"] = rating(program["program"])

    return create_json_data(ranked)
