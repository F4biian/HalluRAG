import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
import pandas as pd
import json

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pprint import pprint

from hallu_clf import HallucinationClassifier, train_model, test_model, load_checkpoint, print_confusion_matrix, save_confusion_matrix_plot


CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(os.path.join(CURR_DIR, ".."), "data")
INTERNAL_STATES_DIR = os.path.join(os.path.join(DATA_DIR, "RAGTruth"), "internal_states")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_FILE = os.path.join(CURR_DIR, "checkpoint.pth")
RESULTS_FILE = os.path.join(CURR_DIR, "baseline_results.json")

INTERNAL_STATE_NAMES = ['layer_50_last_token', 'layer_100_last_token', 'activations_layer_50_last_token', 'activations_layer_100_last_token'] # 'probability', 'entropy'
MODEL_NAME_STARTS = {
    "Llama-2-7b-chat-hf": {
        "All": "meta-llama_Llama-2-7b-chat-hf",
        "None": "meta-llama_Llama-2-7b-chat-hf.",
        "float8": "meta-llama_Llama-2-7b-chat-hf (float8)",
        "int8": "meta-llama_Llama-2-7b-chat-hf (int8)",
        "int4": "meta-llama_Llama-2-7b-chat-hf (int4)",
    },
    "Llama-2-13b-chat-hf": {
        "All": "meta-llama_Llama-2-13b-chat-hf",
        "None": "meta-llama_Llama-2-13b-chat-hf.",
        "float8": "meta-llama_Llama-2-13b-chat-hf (float8)",
        "int8": "meta-llama_Llama-2-13b-chat-hf (int8)",
        "int4": "meta-llama_Llama-2-13b-chat-hf (int4)",
    },
    "Mistral-7B-Instruct-v0.1": {
        "All": "mistralai_Mistral-7B-Instruct-v0.1",
        "None": "mistralai_Mistral-7B-Instruct-v0.1.",
        "float8": "mistralai_Mistral-7B-Instruct-v0.1 (float8)",
        "int8": "mistralai_Mistral-7B-Instruct-v0.1 (int8)",
        "int4": "mistralai_Mistral-7B-Instruct-v0.1 (int4)",
    }
}

np.random.seed(432)

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

def train_val_test_split(X, y, source_ids, val_size=0.15, test_size=0.15, correct_imbalance=True, oversampling=False):
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
    if correct_imbalance:
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
    if correct_imbalance:
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
    if correct_imbalance:
        print("Rebalancing test data based on target...")
        X_test, y_test, _ = correct_binary_imbalance(X_test, y_test, source_ids[test_indices], oversampling=oversampling)
        print("Rebalanced.")
        print(f"X_test.shape: {X_test.shape}")
        print(f"y_test.shape: {y_test.shape}")
        print(f"y_test.mean(): {y_test.mean()}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_data(model_name, internal_states_name, val_size=0.15, test_size=0.15, correct_imbalance=True, oversampling=False):
    X = []
    y = []
    source_ids = []

    for file_name in os.listdir(INTERNAL_STATES_DIR):
        if file_name.startswith(model_name):
            print(f"Adding data of file {file_name} to data...")
            with open(os.path.join(INTERNAL_STATES_DIR, file_name), 'rb') as handle:
                file_json = pickle.load(handle)
                for passage_data in file_json:
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

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, source_ids, val_size=val_size, test_size=test_size, correct_imbalance=correct_imbalance, oversampling=oversampling)

    total_data_count = y_train.shape[0] + y_val.shape[0] + y_test.shape[0]
    print(f"Final train size: {round(y_train.shape[0] * 100 / total_data_count, 4)}%")
    print(f" Final test size: {round(y_val.shape[0] * 100 / total_data_count, 4)}%")
    print(f"  Final val size: {round(y_test.shape[0] * 100 / total_data_count, 4)}%")

    return X_train, X_val, X_test, y_train, y_val, y_test

def run(model_name, internal_states_name, runs=15):
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(model_name, internal_states_name, val_size=0.15, test_size=0.15, correct_imbalance=True, oversampling=True)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

    run_results = []

    for run_i in range(runs):
        # Defining model, loss and optimizer
        model = HallucinationClassifier(X_train.shape[1], dropout_p=0.5).to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train model
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            checkpoint_file=CHECKPOINT_FILE,
            epochs=100,
            stop_when_not_improved_after=5
        )

        train_loss, train_acc, train_precision, train_recall, train_f1, train_fpr, train_tpr, train_roc_auc, train_P, train_R, train_auc_pr, train_conf_matrix = test_model(model, train_loader, criterion, None)
        val_loss, val_acc, val_precision, val_recall, val_f1, val_fpr, val_tpr, val_roc_auc, val_P, val_R, val_auc_pr, val_conf_matrix = test_model(model, val_loader, criterion, None)

        # Load best checkpoint
        load_checkpoint(CHECKPOINT_FILE, model, optimizer)

        test_loss, test_acc, test_precision, test_recall, test_f1, test_fpr, test_tpr, test_roc_auc, test_P, test_R, test_auc_pr, test_conf_matrix = test_model(model, test_loader, criterion, None)

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
            "train": {
                "loss": train_loss,
                "acc": train_acc,
                "p": train_precision,
                "r": train_recall,
                "f1": train_f1,
                "fpr": train_fpr.tolist(),
                "tpr": train_tpr.tolist(),
                "roc_auc": train_roc_auc,
                "P": train_P.tolist(),
                "R": train_R.tolist(),
                "auc_pr": train_auc_pr,
                "conf_matrix": train_conf_matrix.tolist()
            },
            "val": {
                "loss": val_loss,
                "acc": val_acc,
                "p": val_precision,
                "r": val_recall,
                "f1": val_f1,
                "fpr": val_fpr.tolist(),
                "tpr": val_tpr.tolist(),
                "auc": val_roc_auc,
                "P": val_P.tolist(),
                "R": val_R.tolist(),
                "auc_pr": val_auc_pr,
                "conf_matrix": val_conf_matrix.tolist()
            },
            "test": {
                "loss": test_loss,
                "acc": test_acc,
                "p": test_precision,
                "r": test_recall,
                "f1": test_f1,
                "fpr": test_fpr.tolist(),
                "tpr": test_tpr.tolist(),
                "auc": test_roc_auc,
                "P": test_P.tolist(),
                "R": test_R.tolist(),
                "auc_pr": test_auc_pr,
                "conf_matrix": test_conf_matrix.tolist()
            }
        })

        # print("-"*30)
        # print(f"Test Loss:\t{test_loss}")
        # print(f"Test Accuracy:\t{test_acc}")
        # print(f"Test Precision:\t{test_precision}")
        # print(f"Test Recall:\t{test_recall}")
        # print(f"Test AUC:\t{test_auc}")
        # print("-"*30)
        # print_confusion_matrix(test_conf_matrix)
        # print("-"*30)
        # save_confusion_matrix_plot(test_conf_matrix, CONF_MATRIX_FILE)
    return run_results

if __name__ == "__main__":
    model_results = {}

    for model_name, model_name_starts_dict in MODEL_NAME_STARTS.items():
        quant_results = {}
        for quant_name, model_name_start in model_name_starts_dict.items():
            internal_states_results = {}
            for internal_state_name in INTERNAL_STATE_NAMES:
                internal_states_results[internal_state_name] = run(model_name_start, internal_state_name)

            quant_results[quant_name] = internal_states_results
        model_results[model_name] = quant_results

    with open(RESULTS_FILE, "w") as file:
        json.dump(model_results, file)