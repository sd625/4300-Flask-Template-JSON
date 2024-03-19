import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import numpy as np
import re

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'cornell-programs')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)[0]
    programs = pd.DataFrame(data['programs'])

app = Flask(__name__)
CORS(app)

# def json_search(query):
#     matches = []
#     merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
#     matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
#     matches_filtered = matches[['title', 'descr', 'imdb_rating']]
#     matches_filtered_json = matches_filtered.to_json(orient='records')
#     return matches_filtered_json


def jaccard(s1, s2):
    if len(s1) == 0 or len(2) == 0:
        return 0 
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    jaccard = intersection / union
    return jaccard

def tokenize(text):
    text = text.lower()
    words = re.findall("[a-zA-Z]+", text)
    
    return words
    
def rank_programs_jaccard(data, query):
    query_words = tokenize(query)
    rankings = []
    for program in data:
        program_name_words = tokenize(program['program_name'])
        similarity = jaccard(query_words, program_name_words)
        rankings.append((program['id'], similarity))
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/search")
def search():
    text = request.args.get("title")
    return rank_programs_jaccard(text)

# @app.route("/episodes")
# def episodes_search():
#     text = request.args.get("title")
#     return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)