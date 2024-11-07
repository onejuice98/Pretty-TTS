import os
import json


def get_file_list(path, extension="json"):
    file_list = os.listdir(f"{path}")
    file_list = [file for file in file_list if file.endswith(f".{extension}")]

    return file_list

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)