import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import numpy as np
import re

from svd import *
from sentiment_rank import sentiment_ranking
from similarity import compute_composite_similarity, compute_cosine_similarity

# number of results on one page
MAX_RESULTS = 20

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ["ROOT_PATH"] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, "cornell-programs-tokens.json")

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, "r") as file:
    data = json.load(file)
    programs_df = pd.DataFrame(data["programs"])

app = Flask(__name__)
CORS(app)


def insertion_cost(message, j):
    return 1


def deletion_cost(query, i):
    return 1


def substitution_cost(query, message, i, j):
    if query[i - 1] == message[j - 1]:
        return 0
    else:
        return 1


def edit_matrix_func(query, message, ins_cost_func, del_cost_func, sub_cost_func):

    m = len(query) + 1
    n = len(message) + 1

    chart = {(0, 0): 0}
    for i in range(1, m):
        chart[i, 0] = chart[i - 1, 0] + del_cost_func(query, i)
    for j in range(1, n):
        chart[0, j] = chart[0, j - 1] + ins_cost_func(message, j)
    for i in range(1, m):
        for j in range(1, n):
            chart[i, j] = min(
                chart[i - 1, j] + del_cost_func(query, i),
                chart[i, j - 1] + ins_cost_func(message, j),
                chart[i - 1, j - 1] + sub_cost_func(query, message, i, j),
            )
    return chart


def edit_distance(
    query: str, message: str, ins_cost_func: int, del_cost_func: int, sub_cost_func: int
) -> int:

    query = query.lower()
    message = message.lower()

    edit_matrix = edit_matrix_func(
        query, message, ins_cost_func, del_cost_func, sub_cost_func
    )

    edit_distance = edit_matrix[(len(query), len(message))]

    return edit_distance


def rank_program_results(
    query: str,
):
    rankings = []

    query_tokens = tokenize(query)

    for index, row in programs_df.iterrows():

        program_name = row["program"]
        program_name_tokens = tokenize(program_name)

        program_location = row["location"]
        program_location_tokens = tokenize(program_location)

        program_tokens = row["tokens"]

        program_url = row["url"]
        program_gpa = row["gpa"]
        program_colleges = row["colleges"]

        # if query is empty, return programs in alphabetical order
        if query == "":
            rankings.append(
                (
                    index,
                    -1,
                    program_name,
                    program_location,
                    program_url,
                    program_gpa,
                    program_colleges,
                )
            )
        else:
            name_edit_distances = [
                edit_distance(
                    query_token,
                    program_token,
                    insertion_cost,
                    deletion_cost,
                    substitution_cost,
                )
                for query_token in query_tokens
                for program_token in program_name_tokens
            ]
            name_edit_distance = min(name_edit_distances)

            max_name_distance = max(len(query_tokens), len(program_name_tokens))
            normalized_name_edit_distance = name_edit_distance / max_name_distance
            name_jaccard = jaccard(query_tokens, program_name_tokens)
            name_score = max((1 - normalized_name_edit_distance), name_jaccard)

            location_edit_distances = [
                edit_distance(
                    query_token,
                    location_token,
                    insertion_cost,
                    deletion_cost,
                    substitution_cost,
                )
                for query_token in query_tokens
                for location_token in program_location_tokens
            ]
            location_edit_distance = min(location_edit_distances)

            location_jaccard = jaccard(query_tokens, program_location_tokens)

            max_location_distance = max(len(query_tokens), len(program_location_tokens))
            normalized_location_edit_distance = (
                location_edit_distance / max_location_distance
            )
            location_score = max(
                (1 - normalized_location_edit_distance), location_jaccard
            )

            token_edit_distances = [
                edit_distance(
                    query_token,
                    program_token,
                    insertion_cost,
                    deletion_cost,
                    substitution_cost,
                )
                for query_token in query_tokens
                for program_token in program_tokens
            ]

            max_token_distance = max(len(query_tokens), len(program_tokens))

            if token_edit_distances:
                token_edit_distance = min(token_edit_distances)
            else:
                token_edit_distance = max_token_distance

            normalized_token_edit_distance = token_edit_distance / max_token_distance
            token_score = 1 - normalized_token_edit_distance

            name_weight = 0.4
            location_weight = 0.55
            token_weight = 0.05

            score = (
                name_weight * name_score
                + location_weight * location_score
                + token_weight * token_score
            )

            program_info = program_name_tokens.union(
                program_location_tokens, program_tokens
            )
            combo = compute_composite_similarity(query, program_info, score)
            score = combo

            rankings.append(
                (
                    index,
                    score,
                    program_name,
                    program_location,
                    program_url,
                    program_gpa,
                    program_colleges,
                )
            )

    rankings.sort(key=lambda x: x[1], reverse=True)
    l = min(len(rankings), MAX_RESULTS)
    rankings = rankings[:l]
    sim_ranked = [i for i in rankings if i[1] >= 0.5]
    sim_ranked.sort(key=lambda x: x[1], reverse=True)
    if sim_ranked != []:
        rankings = sim_ranked

    json_data = [
        {
            "id": id,
            "program": program_name,
            "program_location": program_location,
            "url": program_url,
            "gpa": program_gpa,
            "colleges": program_colleges,
        }
        for id, _, program_name, program_location, program_url, program_gpa, program_colleges in rankings
    ]
    return json_data


def jaccard(s1, s2):
    # s1 = tokenize(s1)
    # s2 = tokenize(s2)
    if len(s1) == 0 or len(s2) == 0:
        return 0
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    jaccard = intersection / union
    return jaccard


def tokenize(text):
    text = text.lower()
    words = re.findall("[a-zA-Z]+", text)
    return set(words)


def rank_programs_jaccard(query):
    query_words = tokenize(query)
    rankings = []
    for index, row in programs_df.iterrows():

        program_name = row["program"]
        program_location = row["location"]
        program_tokens = row["tokens"]

        program_url = row["url"]
        program_gpa = row["gpa"]
        program_colleges = row["colleges"]

        program_info = tokenize(program_name).union(
            tokenize(" ".join(program_tokens)), tokenize(program_location)
        )
        similarity = jaccard(query_words, program_info)
        if query == "":
            rankings.append(
                (
                    index,
                    -1,
                    program_name,
                    program_location,
                    program_url,
                    program_gpa,
                    program_colleges,
                )
            )
        else:
            if similarity > 0:
                rankings.append(
                    (
                        index,
                        similarity,
                        program_name,
                        program_location,
                        program_url,
                        program_gpa,
                        program_colleges,
                    )
                )

    rankings.sort(key=lambda x: x[1], reverse=True)
    l = min(len(rankings), MAX_RESULTS)
    rankings = rankings[:l]
    sim_ranked = [i for i in rankings if i[1] >= 0.5]
    sim_ranked.sort(key=lambda x: x[1], reverse=True)
    if sim_ranked != []:
        rankings = sim_ranked

    json_data = [
        {
            "id": id,
            "program": program_name,
            "program_location": program_location,
            "url": program_url,
            "gpa": program_gpa,
            "colleges": program_colleges,
        }
        for id, _, program_name, program_location, program_url, program_gpa, program_colleges in rankings
    ]

    return json_data


def filtering(search, gpa="", college="", location="", flexible="true"):
    filtered_list = []
    for program in search:
        sims = []

        if location != "":
            # print("location was entered")
            # print(type(location))
            loc_filter_toks = tokenize(location)
            prog_loc_toks = tokenize(program["program_location"])
            loc_sim = jaccard(prog_loc_toks, loc_filter_toks)
            sims.append(loc_sim)

            # note: issue with united kingdom for some reason, figure out why

        if college != "":
            college_filter_toks = tokenize(college)
            prog_college_toks = tokenize(program["colleges"])
            college_sim = jaccard(prog_college_toks, college_filter_toks)
            sims.append(college_sim)

        if gpa != "":
            gpa_lower_bound = float(gpa[0:3])
            gpa_upper_bound = float(gpa[4:])
            prog_gpa = float(program["gpa"])
            gpa_sim = 0
            if prog_gpa != -1:
                if gpa_lower_bound <= prog_gpa <= gpa_upper_bound:
                    gpa_sim = 1
            sims.append(gpa_sim)

        # if department != "":
        #     dept__filter_toks = tokenize(department)
        #     prog_dept_toks = tokenize(program['department'])
        #     dept_sim = jaccard(prog_dept_toks, dept__filter_toks)
        #     sims.append(dept_sim)

        # checked = true, so when flexible
        # false = not flexible = unchecked
        if flexible == "true":
            # print("flexible", flexible)
            if any(sims):
                filtered_list.append(program)
        if flexible == "false":
            # print("not flexible")
            if all(sims):
                filtered_list.append(program)
        # print(sims)
        # print("flexible?", any(sims))
        # print("not flexible?", all(sims))

    if len(filtered_list) == 0:
        filtered_list = search

    json_string = json.dumps(filtered_list, indent=2)
    return json_string


@app.route("/")
def home():
    return render_template("base.html", title="sample html")


@app.route("/search_programs")
def search():
    text = request.args.get("title")
    min_gpa = request.args.get("gpa")
    college = request.args.get("college")
    location = request.args.get("location")
    flexible = request.args.get("flexible")
    rank = rank_programs_jaccard(text)
    sent_rank = sentiment_ranking(text, rank)
    filtered = filtering(sent_rank, min_gpa, college, location, flexible)
    return filtered


if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
