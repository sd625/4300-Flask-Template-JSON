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
json_file_path = os.path.join(current_directory, 'cornell-programs.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    programs_df = pd.DataFrame(data['programs'])

app = Flask(__name__)
CORS(app)

def jaccard(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    if len(s1) == 0 or len(s2) == 0:
        return 0 
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    jaccard = intersection / union
    return jaccard

def tokenize(text):
    text = text.lower()
    words = re.findall("[a-zA-Z]+", text)
    
    return words
    
def rank_programs_jaccard(query):
    query_words = tokenize(query)
    rankings = []
    counter = 0
    for index, row in programs_df.iterrows():
        
        program_name = row['program_name']
        program_location = row['location']
        program_name_words = tokenize(program_name)
        program_location_words = tokenize(program_location)
        program_info = program_name_words + program_location_words
        similarity = jaccard(query_words, program_info)
        rankings.append((counter, similarity, program_name, program_location))
        counter += 1
    rankings.sort(key=lambda x: x[1], reverse=True)
    rankings = rankings[:10]
    json_data = [
    {"id": id, "program_name": program_name, "program_location": program_location}
    for id, _, program_name, program_location in rankings]
    print(json_data)

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