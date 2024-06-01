import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
import pandas as pd
import json
import random

from torch.utils.data import DataLoader, TensorDataset
from pprint import pprint

from hallu_clf import HallucinationClassifier, train_model, test_model, load_checkpoint, get_random_classifier_results,get_thresolds_from_results


CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(os.path.join(CURR_DIR, ".."), "data")
LONG_PROMPT_INTERNAL_STATES_DIR = os.path.join(os.path.join(DATA_DIR, "RAGTruth"), "long_prompt_internal_states")
INTERNAL_STATES_DIR = os.path.join(os.path.join(DATA_DIR, "RAGTruth"), "internal_states")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_FILE = os.path.join(CURR_DIR, "checkpoint_long_prompt.pth")
RESULTS_FILE = os.path.join(CURR_DIR, "baseline_results_long_prompt.json")

LONG_FILE = os.path.join(LONG_PROMPT_INTERNAL_STATES_DIR, "mistralai_Mistral-7B-Instruct-v0.1 (float8).pickle")
SHORT_FILE = os.path.join(INTERNAL_STATES_DIR, "mistralai_Mistral-7B-Instruct-v0.1 (float8).pickle")
LONG_SHORT_THRESHOLD = 2658 # max prompt length for short prompts and min length for long ones (-> 609 long and 609 short prompts remain)  

INTERNAL_STATE_NAMES = ['layer_50_last_token', 'layer_100_last_token', 'activations_layer_50_last_token', 'activations_layer_100_last_token'] # 'probability', 'entropy'
FILE_COMBINATIONS = {
    "Both-Both": {
        "train": [LONG_FILE, SHORT_FILE],
        "test": [LONG_FILE, SHORT_FILE]
    },
    "Short-Both": {
        "train": [SHORT_FILE],
        "test": [LONG_FILE, SHORT_FILE]
    },
    "Long-Both": {
        "train": [LONG_FILE],
        "test": [LONG_FILE, SHORT_FILE]
    }
}

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(432)

def correct_binary_imbalance(X, y, source_ids, oversampling=False):
    classes1 = np.sum(y)
    classes0 = y.shape[0] - classes1

    indices_class1 = np.where(y)[0]
    indices_class0 = np.where(~y)[0]

    if classes1 > classes0:
        if oversampling:
            while indices_class0.shape[0] < indices_class1.shape[0]:
                new_indices0 = np.random.choice(indices_class0, size=min(indices_class0.shape[0], indices_class1.shape[0] - indices_class0.shape[0]), replace=False)
                indices_class0 = np.concatenate([indices_class0, new_indices0])
        else:
            indices_class1 = indices_class1[:classes0]
    elif classes0 > classes1:
        if oversampling:
            while indices_class1.shape[0] < indices_class0.shape[0]:
                new_indices1 = np.random.choice(indices_class1, size=min(indices_class1.shape[0], indices_class0.shape[0] - indices_class1.shape[0]), replace=False)
                indices_class1 = np.concatenate([indices_class1, new_indices1])
        else:
            indices_class0 = indices_class0[:classes1]

    all_indices = np.concatenate([indices_class0, indices_class1])
    np.random.shuffle(all_indices)

    return X[all_indices], y[all_indices], source_ids[all_indices]

def train_val_test_split(X, y, source_ids, val_size=0.15, test_size=0.15, correct_imbalance_train=True, correct_imbalance_val=False, correct_imbalance_test=False, oversampling=False):
    ids_sorted = pd.Series(source_ids).sort_values()

    border_id = ids_sorted.iloc[int(ids_sorted.shape[0]*(1-val_size-test_size))]
    train_val_border = np.where(ids_sorted == border_id)[0][0]
    train_indices = np.array(ids_sorted.iloc[:train_val_border].index)

    border_id = ids_sorted.iloc[int(ids_sorted.shape[0]*(1-test_size))]
    val_test_border = np.where(ids_sorted == border_id)[0][0]
    val_indices = np.array(ids_sorted.iloc[train_val_border:val_test_border].index)

    test_indices = np.array(ids_sorted.iloc[val_test_border:].index)

    print()

    X_train = X[train_indices]
    y_train = y[train_indices]
    print(f"X_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"y_train.mean(): {y_train.mean()}")
    if correct_imbalance_train:
        print("Rebalancing train data based on target...")
        X_train, y_train, _ = correct_binary_imbalance(X_train, y_train, source_ids[train_indices], oversampling=oversampling)
        print("Rebalanced.")
        print(f"X_train.shape: {X_train.shape}")
        print(f"y_train.shape: {y_train.shape}")
        print(f"y_train.mean(): {y_train.mean()}")

    print()

    X_val = X[val_indices]
    y_val = y[val_indices]
    print(f"X_val.shape: {X_val.shape}")
    print(f"y_val.shape: {y_val.shape}")
    print(f"y_val.mean(): {y_val.mean()}")
    if correct_imbalance_val:
        print("Rebalancing val data based on target...")
        X_val, y_val, _ = correct_binary_imbalance(X_val, y_val, source_ids[val_indices], oversampling=oversampling)
        print("Rebalanced.")
        print(f"X_val.shape: {X_val.shape}")
        print(f"y_val.shape: {y_val.shape}")
        print(f"y_val.mean(): {y_val.mean()}")

    print()

    X_test = X[test_indices]
    y_test = y[test_indices]
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_test.shape: {y_test.shape}")
    print(f"y_test.mean(): {y_test.mean()}")
    if correct_imbalance_test:
        print("Rebalancing test data based on target...")
        X_test, y_test, _ = correct_binary_imbalance(X_test, y_test, source_ids[test_indices], oversampling=oversampling)
        print("Rebalanced.")
        print(f"X_test.shape: {X_test.shape}")
        print(f"y_test.shape: {y_test.shape}")
        print(f"y_test.mean(): {y_test.mean()}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_data(files, internal_states_name, val_size=0.15, test_size=0.15, correct_imbalance_train=True, correct_imbalance_val=False, correct_imbalance_test=False, oversampling=False):
    X = []
    y = []
    source_ids = []

    for file_name in files:
        print(f"Adding data of file {file_name} to data...")
        with open(file_name, 'rb') as handle:
            file_json = pickle.load(handle)
            for passage_data in file_json:

                # Skip this data point of prompt too short/long
                if file_name == LONG_FILE:
                    if len(passage_data["prompt"]) < LONG_SHORT_THRESHOLD: # too short "long prompt"
                        continue
                if file_name == SHORT_FILE:
                    if len(passage_data["prompt"]) >= LONG_SHORT_THRESHOLD: # too long "short prompt"
                        continue

                for sentence_data in passage_data["sentence_data"]:
                    target = sentence_data["target"]

                    # 'layer_50_last_token', 'layer_100_last_token', 'activations_layer_50_last_token', 'activations_layer_100_last_token', 'probability', 'entropy' as keys
                    internal_states = sentence_data["internal_states"][internal_states_name]
                    
                    X.append(internal_states)
                    y.append(target)
                    source_ids.append(passage_data["source_id"])

    X = np.array(X)
    y = np.array(y, dtype=bool)
    source_ids = np.array(source_ids)
        
    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")
    print(f"y.mean(): {y.mean()}")

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, source_ids, val_size=val_size, test_size=test_size, correct_imbalance_train=correct_imbalance_train, correct_imbalance_val=correct_imbalance_val, correct_imbalance_test=correct_imbalance_test, oversampling=oversampling)

    total_data_count = y_train.shape[0] + y_val.shape[0] + y_test.shape[0]
    print(f"Final train size: {round(y_train.shape[0] * 100 / total_data_count, 4)}%")
    print(f" Final test size: {round(y_val.shape[0] * 100 / total_data_count, 4)}%")
    print(f"  Final val size: {round(y_test.shape[0] * 100 / total_data_count, 4)}%")

    return X_train, X_val, X_test, y_train, y_val, y_test

def run(train_files, test_files, internal_states_name, runs=10):
    long_X_train, long_X_val, long_X_test, long_y_train, long_y_val, long_y_test = get_data([LONG_FILE], internal_states_name, val_size=0.15, test_size=0.15, correct_imbalance_train=True, correct_imbalance_val=True, correct_imbalance_test=True, oversampling=True)
    short_X_train, short_X_val, short_X_test, short_y_train, short_y_val, short_y_test = get_data([SHORT_FILE], internal_states_name, val_size=0.15, test_size=0.15, correct_imbalance_train=True, correct_imbalance_val=True, correct_imbalance_test=True, oversampling=True)

    if SHORT_FILE in train_files and LONG_FILE in train_files:
        train_min = min(long_X_train.shape[0], short_X_train.shape[0])
        long_X_train = long_X_train[:train_min]
        long_y_train = long_y_train[:train_min]

        val_min = min(long_X_val.shape[0], short_X_val.shape[0])
        long_X_val = long_X_val[:val_min]
        long_y_val = long_y_val[:val_min]
        
        X_train = np.concatenate([long_X_train, short_X_train])
        y_train = np.concatenate([long_y_train, short_y_train])
        X_val = np.concatenate([long_X_val, short_X_val])
        y_val = np.concatenate([long_y_val, short_y_val])
    elif SHORT_FILE in train_files:
        X_train = short_X_train
        y_train = short_y_train
        X_val = short_X_val
        y_val = short_y_val
    elif LONG_FILE in train_files:
        X_train = long_X_train
        y_train = long_y_train
        X_val = long_X_val
        y_val = long_y_val
    else:
        raise Exception()

    if SHORT_FILE in test_files and LONG_FILE in test_files:
        test_min = min(long_X_test.shape[0], short_X_test.shape[0])
        long_X_test = long_X_test[:test_min]
        long_y_test = long_y_test[:test_min]

        X_test = np.concatenate([long_X_test, short_X_test])
        y_test = np.concatenate([long_y_test, short_y_test])
    elif SHORT_FILE in test_files:
        X_test = short_X_test
        y_test = short_y_test
    elif LONG_FILE in test_files:
        X_test = long_X_test
        y_test = long_y_test
    else:
        raise Exception()

    print("After composition: X_train.shape:", X_train.shape)
    print("After composition: X_val.shape:", X_val.shape)
    print("After composition: X_test.shape:", X_test.shape)
    print("After composition: y_train.mean():", y_train.mean())
    print("After composition: y_val.mean():", y_val.mean())
    print("After composition: y_test.mean():", y_test.mean())

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=16)

    run_results = []

    for run_i in range(runs):
        # Defining model, loss and optimizer
        model = HallucinationClassifier(X_train.shape[1], dropout_p=0.15).to(DEVICE)

        # class_weight_no_hallucination = y_train.shape[0] / ((y_train.shape[0] - y_train.sum()) * 2) # class 0
        # class_weight_hallucination = y_train.shape[0] / (y_train.sum() * 2) # class 1
        # imbalance_weights = torch.FloatTensor([class_weight_hallucination]).to(DEVICE)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0000025, weight_decay=1e-5)

        test_results_random = get_random_classifier_results(test_loader)
        # print(test_results_random["auc_pr_grounded"])
        # print(test_results_random["auc_pr_hallucinated"])

        # Train model
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            checkpoint_file=CHECKPOINT_FILE,
            epochs=500,
            stop_when_not_improved_after=20
        )

        # Load best checkpoint
        load_checkpoint(CHECKPOINT_FILE, model, optimizer)

        train_results = test_model(model, train_loader, criterion)
        thresholds = get_thresolds_from_results(train_results)
        val_results = test_model(model, val_loader, criterion, thresholds)

        test_results = test_model(model, test_loader, criterion, thresholds)
        # print(test_results["auc_pr_grounded"])
        # print(test_results["auc_pr_hallucinated"])
        # exit()

        run_results.append({
            "X_train.size": X_train.shape,
            "y_train.size": y_train.shape,
            "y_train.mean": y_train.mean(),
            "X_val.size": X_val.shape,
            "y_val.size": y_val.shape,
            "y_val.mean": y_val.mean(),
            "X_test.size": X_test.shape,
            "y_test.size": y_test.shape,
            "y_test.mean": y_test.mean(),
            "i": run_i,
            "train": train_results,
            "val": val_results,
            "test": test_results,
            "test_random": test_results_random
        })
    return run_results

if __name__ == "__main__":
    model_results = {}

    for model_name, _ in [("Mistral-7B-Instruct-v0.1 (float8)", None)]:
        file_comb_results = {}
        for file_comb_name, file_comb_dict in FILE_COMBINATIONS.items():
            internal_states_results = {}
            for internal_state_name in INTERNAL_STATE_NAMES:
                internal_states_results[internal_state_name] = run(file_comb_dict["train"], file_comb_dict["test"], internal_state_name)

            file_comb_results[file_comb_name] = internal_states_results
        model_results[model_name] = file_comb_results

    with open(RESULTS_FILE, "w") as file:
        json.dump(model_results, file, default=str)