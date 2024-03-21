import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import numpy as np
import re

#number of results on one page
MAX_RESULTS = 10

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'cornell-programs.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    programs_df = pd.DataFrame(data['programs'])

app = Flask(__name__)
CORS(app)

def jaccard(s1, s2):
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
        
        program_name = row['program_name']
        program_location = row['location']
        
        program_info = tokenize(program_name).union(tokenize(program_location))
        similarity = jaccard(query_words, program_info)

        if similarity > 0:
            rankings.append((index, similarity, program_name, program_location))
        
    rankings.sort(key=lambda x: x[1], reverse=True)
    l = min(len(rankings), MAX_RESULTS)
    rankings = rankings[:l]

    json_data = [
    {"id": id, "program_name": program_name, "program_location": program_location}
    for id, _, program_name, program_location in rankings]

    json_string = json.dumps(json_data, indent=2)

    return json_string


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/search_programs")
def search():
    text = request.args.get("title")
    return rank_programs_jaccard(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)