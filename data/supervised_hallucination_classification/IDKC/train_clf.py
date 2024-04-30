########################################################################################
# IMPORTS

import fasttext
import os

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
TRAIN_FILE = os.path.join(CURR_DIR, "idk.train")
TEST_FILE = os.path.join(CURR_DIR, "idk.test")
MODEL_FILE = os.path.join(CURR_DIR, "idk_model.bin")
TRAIN_PERC = 0.75

def create_files() -> None:
    # TODO: implement once data exist
    # str.lower()
    # str.replace('^[\w\s]', '', regex=True) # remove punctuation
    # str.replace('\d', '', regex=True) # remove digits
    # str.replace('\s', ' ', regex=True) # replace every form of whitespace with a space
    # remove stopwords wit nltk?
    # add either __label__idk or __label__ik before each sentence (one space between label and sentence)
    # split into train and test
    # save files
    raise NotImplementedError()

create_files()

# Train model
model = fasttext.train_supervised(input=TRAIN_FILE) # , lr=1.0, epoch=25, wordNgrams=2, dim=50

# Show train results
train_results = model.test(TRAIN_FILE)
print(f"Train Results: {train_results}")

# Test model
test_results = model.test(TEST_FILE)
print(f"Test Results: {test_results}")

# Save model
if input(f"Save model to {MODEL_FILE}? [y|anything else for no]").lower().strip() == "y":
    model.save_model(MODEL_FILE)
    print("Saved!")