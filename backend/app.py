import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import numpy as np
import re

from svd import *

#number of results on one page
MAX_RESULTS = 20

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'cornell-programs-tokens.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    programs_df = pd.DataFrame(data['programs'])

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

    edit_matrix = edit_matrix_func(query, message, ins_cost_func, del_cost_func, sub_cost_func)
      
    edit_distance = edit_matrix[(len(query),len(message))]

    return edit_distance

def rank_program_results(
    query: str,
):
    rankings = []

    query_tokens = tokenize(query)
    
    for index, row in programs_df.iterrows():
        
        program_name = row['program']
        program_name_tokens = tokenize(program_name)

        program_location = row['location']
        program_location_tokens = tokenize(program_location)
        
        program_tokens = row['tokens']

        program_url = row['url']

        name_edit_distances = [
            edit_distance(query_token, program_token, insertion_cost, deletion_cost, substitution_cost)
            for query_token in query_tokens
            for program_token in program_name_tokens
        ]
        name_edit_distance = min(name_edit_distances)

        max_name_distance = max(len(query_tokens), len(program_name_tokens))
        normalized_name_edit_distance = name_edit_distance / max_name_distance

        name_jaccard = jaccard(query, program_name)
        name_score = max((1-normalized_name_edit_distance), name_jaccard)

        location_edit_distances = [
            edit_distance(query_token, location_token, insertion_cost, deletion_cost, substitution_cost)
            for query_token in query_tokens
            for location_token in program_location_tokens
        ]
        location_edit_distance = min(location_edit_distances)
        
        location_jaccard = jaccard(query, program_location)

        max_location_distance = max(len(query_tokens), len(program_location_tokens))
        normalized_location_edit_distance = location_edit_distance / max_location_distance
        location_score = max((1-normalized_location_edit_distance), location_jaccard)

        
        token_edit_distances = [
            edit_distance(query_token, program_token, insertion_cost, deletion_cost, substitution_cost)
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

        
        score = (name_weight * name_score +
                 location_weight * location_score + 
                 token_weight * token_score)
        

        print(program_name, name_score, location_score, token_score, score)

        rankings.append((index, score, program_name, program_location, program_url))
        
    rankings.sort(key=lambda x: x[1], reverse=True)
    l = min(len(rankings), MAX_RESULTS)
    rankings = rankings[:l]

    json_data = [
    {"id": id, "program": program_name, "program_location": program_location, "url": program_url}
    for id, score, program_name, program_location, program_url in rankings]

    json_string = json.dumps(json_data, indent=2)

    return json_string
      

def jaccard(s1, s2):
    s1 = tokenize(s1)
    s2 = tokenize(s2)
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
        
        program_name = row['program']
        program_location = row['location']
        
        program_info = tokenize(program_name).union(tokenize(program_location))
        similarity = jaccard(query_words, program_info)

        if similarity > 0:
            rankings.append((index, similarity, program_name, program_location))
        
    rankings.sort(key=lambda x: x[1], reverse=True)
    l = min(len(rankings), MAX_RESULTS)
    rankings = rankings[:l]

    json_data = [
    {"id": id, "program": program_name, "program_location": program_location}
    for id, _, program_name, program_location in rankings]

    json_string = json.dumps(json_data, indent=2)

    return json_string

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/search_programs")
def search():
    text = request.args.get("title")
    return rank_program_results(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
