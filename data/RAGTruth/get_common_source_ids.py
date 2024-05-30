import os
import pickle
import json

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(os.path.join(CURR_DIR, ".."), "data")
INTERNAL_STATES_DIR = os.path.join(CURR_DIR, "internal_states")
SOURCE_IDS_FILE = os.path.join(CURR_DIR, "common_source_ids.json")

if __name__ == "__main__":
    source_ids = []
    prev_source_ids = None
    all_ids = set()

    for file_name in os.listdir(INTERNAL_STATES_DIR):
        print(f"Adding data of file {file_name} to data...")
        with open(os.path.join(INTERNAL_STATES_DIR, file_name), 'rb') as handle:
            file_json = pickle.load(handle)
            print(len(file_json))
            for passage_data in file_json:
                if prev_source_ids is None or passage_data["source_id"] in prev_source_ids:
                    source_ids.append(passage_data["source_id"])
                all_ids.add(passage_data["source_id"])

        prev_source_ids = source_ids.copy()
        source_ids = []
    
    with open(SOURCE_IDS_FILE, "w") as file:
        json.dump(prev_source_ids, file)