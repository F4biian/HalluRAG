import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
import pandas as pd
import json
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

from torch.utils.data import DataLoader, TensorDataset
from pprint import pprint

from hallu_clf import HallucinationClassifier, train_model, test_model, load_checkpoint, get_random_classifier_results,get_thresolds_from_results

 
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(os.path.join(CURR_DIR, ".."), "data")
INTERNAL_STATES_DIR = os.path.join(DATA_DIR, "HalluRAG")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_FILE = os.path.join(CURR_DIR, "checkpoint2.pth")
RESULTS_FILE = os.path.join(CURR_DIR, "hallurag_results_unanswerable.json")

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
        "All Quantizations": "meta-llama_Llama-2-13b-chat-hf",
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

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(432)

def get_shd_prediction(answerable, pred):
    if pred is None:
        return None

    conflicting_fail_content = pred["conflicting_fail_content"]
    conflicting_fail = pred["conflicting_fail"]
    grounded_fail_content = pred["grounded_fail_content"]
    grounded_fail = pred["grounded_fail"]
    no_clear_answer_fail_content = pred["no_clear_answer_fail_content"]
    no_clear_answer_fail = pred["no_clear_answer_fail"]

    has_fail = conflicting_fail_content or conflicting_fail or grounded_fail_content or grounded_fail or no_clear_answer_fail_content or no_clear_answer_fail
    if has_fail:
        return None

    conflicting = pred["conflicting"]
    grounded = pred["grounded"]
    has_factual_information = pred["has_factual_information"]
    no_clear_answer = pred["no_clear_answer"]

    if conflicting is None or grounded is None or has_factual_information is None or no_clear_answer is None:
        return None

    if grounded and has_factual_information and no_clear_answer:
        return None

    if answerable:
        if conflicting:
            prediction = 1
        elif has_factual_information and grounded:
            prediction = 0
        elif no_clear_answer:
            prediction = 1
        else:
            if has_factual_information:
                if grounded:
                    prediction = 0
                else:
                    prediction = 1
            else:
                prediction = 0
    else:
        if conflicting:
            prediction = 1
        elif not grounded and has_factual_information:
            prediction = 1
        elif no_clear_answer:
            prediction = 0
        else:
            if has_factual_information:
                if grounded:
                    # print("-"*10)
                    # pprint(pred["llm_eval"])
                    # print("-"*10)
                    return None
                    # raise Exception("Unanswerable question has been answered. Should not be possible!")
                else:
                    prediction = 1
            else:
                prediction = 0
    return prediction

def data_disbtr(traits):
    vc = {}
    df = pd.DataFrame(traits)
    for col in df.columns:
        vc[col] = df[col].value_counts(dropna=False).to_dict()
    return vc

def oversample_responses(df_data, X, y, columns=["answerable", "chunk_size", "chunks_per_prompt", "prompt_template_name"], max_iter=10000, no_improv_of=0.005, no_improv_after=100, max_err=0.01) -> pd.DataFrame:
    df = pd.DataFrame(df_data)
    df["i"] = np.arange(df.shape[0], dtype=int)
    orig_df = df.copy()

    unique_combos = df.drop_duplicates(columns, keep="first")
    combo_reproduced_amount = {}
    prev_delete_name = None

    def get_imbalance(df) -> float:
        err = 0
        for col in columns:
            # err += (df[col].value_counts(dropna=False) / df.shape[0]).std()
            rel_vs = (df[col].value_counts(dropna=False) / df.shape[0])
            err += (rel_vs.max() - rel_vs.min()) ** 2
        return err
    
    def find_best_combo_to_delete():
        nonlocal combo_reproduced_amount

        start_err = get_imbalance(df)
        best_ind = None
        best_err_delta = np.inf
        for ind, row in unique_combos.iterrows():
            combo_name = str(row[columns].to_dict())
            if combo_reproduced_amount.get(combo_name, 0) <= 0:
                continue
            if combo_name == str(df[columns].iloc[-1].to_dict()):
                continue
            combo_reproduced_amount[combo_name] = combo_reproduced_amount.get(combo_name, 0) + 1

            pot_df = df.drop(ind) # pd.concat([df, pd.DataFrame(row).T], ignore_index=True)
            new_err = get_imbalance(pot_df)
            err_delta = new_err - start_err
            if err_delta < best_err_delta:
                best_err_delta = err_delta
                best_ind = ind
        if best_ind is not None:
            return best_err_delta, unique_combos.loc[best_ind, columns].to_dict()
        return best_err_delta, None

    def find_best_combo():
        nonlocal prev_delete_name
        start_err = get_imbalance(df)
        best_ind = None
        best_err_delta = np.inf
        for ind, row in unique_combos.iterrows():
            # if str(row[columns].to_dict()) == prev_delete_name:
            #     continue
            pot_df = df._append(row, ignore_index=True) # pd.concat([df, pd.DataFrame(row).T], ignore_index=True)
            new_err = get_imbalance(pot_df)
            err_delta = new_err - start_err
            if err_delta < best_err_delta:
                best_err_delta = err_delta
                best_ind = ind
        if best_ind is not None:
            return best_err_delta, unique_combos.loc[best_ind, columns].to_dict(), unique_combos.loc[best_ind, "i"]
        return best_err_delta, None, None

    err = get_imbalance(df)
    max_iter_counter = 0
    err_hist = [err]
    while err > max_err and max_iter_counter < max_iter:
        best_err_delta, best_combo, best_i = find_best_combo()

        is_delete = False
        # d_best_err_delta, d_best_combo = find_best_combo_to_delete()
        # # print(d_best_err_delta)
        # is_delete = best_err_delta >= 0 or d_best_err_delta < 0 
        # if is_delete:
        #     best_combo = d_best_combo

        best_combo_str = str(best_combo)
        if best_combo_str not in combo_reproduced_amount:
            combo_reproduced_amount[best_combo_str] = 0

        combo_filter = np.ones(shape=df.shape[0], dtype=bool)
        combo_filter[:] = True
        for col in best_combo:
            combo_filter &= (df[col] == best_combo[col]).values
        
        if combo_filter.sum() <= 0:
            combo_filter[df["i"] == best_i] = True

        same_combos = df[combo_filter]
        if is_delete:
            prev_delete_name = best_combo_str
            min_ind = same_combos["i"].value_counts(dropna=False).index[0]
            # print("delete", (df["i"] == min_ind).index[-1], best_combo_str)
            df = df.drop((df["i"] == min_ind).index[-1])
            combo_reproduced_amount[best_combo_str] -= 1
        else:
            try:
                min_ind = same_combos["i"].value_counts(dropna=False).index[-1]
            except:
                print("same_combos", same_combos)
                print("best_combo", best_combo)
                print("columns", columns)
                print("combo_filter", combo_filter.sum())
                raise Exception("myStop")
            # print("add", best_combo_str)
            row_to_reproduce = same_combos.loc[min_ind]
            # df = pd.concat([df, pd.DataFrame(row_to_reproduce).T], ignore_index=True)
            df = df._append(row_to_reproduce, ignore_index=True)
            combo_reproduced_amount[best_combo_str] += 1

        max_iter_counter += 1
        err = get_imbalance(df)
        err_hist.append(err)

        if len(err_hist) >= no_improv_after and err_hist[-no_improv_after] - err <= no_improv_of:
            break

        # print(len(df), err)

    best_i = np.argmin(err_hist) + orig_df.shape[0]
    df = df.iloc[:best_i]
    err = get_imbalance(df)

    new_X = []
    new_y = []
    for i in df["i"]:
        new_X.append(X[i])
        new_y.append(y[i])
    new_X = np.array(new_X)
    new_y = np.array(new_y)

    new_df_data = []
    for _, row in df.iterrows():
        d = row.to_dict()
        del d["i"]
        new_df_data.append(d)

    return err, new_X, new_y, new_df_data

def undersample_responses(df_data, X, y, columns=["answerable", "chunk_size", "chunks_per_prompt", "prompt_template_name"], max_iter=10000, no_improv_of=0.005, no_improv_after=100, max_err=0.01) -> pd.DataFrame:
    df = pd.DataFrame(df_data)
    df["i"] = np.arange(df.shape[0], dtype=int)
    orig_df = df.copy()
    print(df)
    
    seed = 432
    np.random.seed(seed)

    # One-hot encode categorical columns for clustering
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(df[[col for col in df.columns if col != "target" and col != "i"]])

    # Perform KMeans clustering
    num_clusters = min(10, len(df))  # Adjust the number of clusters based on your data size
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
    clusters = kmeans.fit_predict(encoded_features)

    df['cluster'] = clusters

    # Balance classes within each cluster
    balanced_df_list = []

    for cluster in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster]
        df_class_0 = cluster_df[cluster_df['target'] == 0]
        df_class_1 = cluster_df[cluster_df['target'] == 1]
        
        size_minority_class = min(len(df_class_0), len(df_class_1))
        
        if size_minority_class > 0:
            df_class_0_sampled = df_class_0.sample(size_minority_class, random_state=seed)
            df_class_1_sampled = df_class_1.sample(size_minority_class, random_state=seed)
            
            balanced_cluster_df = pd.concat([df_class_0_sampled, df_class_1_sampled])
            balanced_df_list.append(balanced_cluster_df)

    # Combine all balanced clusters
    df_balanced = pd.concat(balanced_df_list)

    # Shuffle the combined DataFrame and drop the cluster column
    df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_balanced = df_balanced.drop(columns=['cluster'])

    df = df_balanced

    # # unique_combos = df.drop_duplicates(columns, keep="first")
    # # combo_reproduced_amount = {}
    # # prev_delete_name = None

    # def get_imbalance(df) -> float:
    #     err = 0
    #     for col in columns:
    #         # err += (df[col].value_counts() / df.shape[0]).std()
    #         rel_vs = (df[col].value_counts() / df.shape[0])
    #         err += (rel_vs.max() - rel_vs.min()) ** 2
    #     return err
    
    # def find_best_combo_to_delete():
    #     # nonlocal combo_reproduced_amount

    #     start_err = get_imbalance(df)
    #     best_ind = None
    #     best_err_delta = np.inf

    #     for ind, row in df.iterrows():
    #         # combo_name = str(row[columns].to_dict())
    #         # if combo_reproduced_amount.get(combo_name, 0) <= 0:
    #         #     continue
    #         # if combo_name == str(df[columns].iloc[-1].to_dict()):
    #         #     continue
    #         # combo_reproduced_amount[combo_name] = combo_reproduced_amount.get(combo_name, 0) + 1

    #         pot_df = df.drop(ind) # pd.concat([df, pd.DataFrame(row).T], ignore_index=True)
    #         new_err = get_imbalance(pot_df)
    #         err_delta = new_err - start_err
    #         if err_delta < best_err_delta:
    #             best_err_delta = err_delta
    #             best_ind = ind
    #     if best_ind is not None:
    #         return best_err_delta, df.loc[best_ind, columns].to_dict()
    #     return best_err_delta, None

    # err = get_imbalance(df)
    # max_iter_counter = 0
    # err_hist = [err]
    # while err > max_err and max_iter_counter < max_iter:
    #     best_err_delta, best_combo = find_best_combo_to_delete()

    #     is_delete = True
    #     # d_best_err_delta, d_best_combo = find_best_combo_to_delete()
    #     # # print(d_best_err_delta)
    #     # is_delete = best_err_delta >= 0 or d_best_err_delta < 0 
    #     # if is_delete:
    #     #     best_combo = d_best_combo

    #     best_combo_str = str(best_combo)
    #     # if best_combo_str not in combo_reproduced_amount:
    #     #     combo_reproduced_amount[best_combo_str] = 0

    #     combo_filter = np.ones(shape=df.shape[0], dtype=bool)
    #     combo_filter[:] = True
    #     for col in best_combo:
    #         combo_filter &= (df.loc[:, col] == best_combo[col])

    #     same_combos = df[combo_filter]
    #     if is_delete:
    #         # prev_delete_name = best_combo_str
    #         max_ind = same_combos["i"].value_counts().index[0]
    #         df = df.drop(df.index[np.where(df["i"] == max_ind)[0][-1]])
    #         # combo_reproduced_amount[best_combo_str] -= 1
    #     else:
    #         min_ind = same_combos["i"].value_counts().index[-1]
    #         # print("add", best_combo_str)
    #         row_to_reproduce = same_combos.loc[min_ind]
    #         # df = pd.concat([df, pd.DataFrame(row_to_reproduce).T], ignore_index=True)
    #         df = df._append(row_to_reproduce, ignore_index=True)
    #         # combo_reproduced_amount[best_combo_str] += 1

    #     max_iter_counter += 1
    #     err = get_imbalance(df)
    #     err_hist.append(err)

    #     if len(err_hist) >= no_improv_after and err_hist[-no_improv_after] - err <= no_improv_of:
    #         break

    #     # print(len(df), err)

    # best_i = np.argmin(err_hist) + orig_df.shape[0]
    # df = df.iloc[:best_i]
    # err = get_imbalance(df)

    new_X = []
    new_y = []
    for i in df["i"]:
        new_X.append(X[i])
        new_y.append(y[i])
    new_X = np.array(new_X)
    new_y = np.array(new_y)

    new_df_data = []
    for _, row in df.iterrows():
        d = row.to_dict()
        del d["i"]
        new_df_data.append(d)

    return 0, new_X, new_y, new_df_data

def get_data(model_name, internal_states_name, correct_imbalance_train=True, correct_imbalance_val=True, correct_imbalance_test=True, oversample=True):
    X_train = []
    y_train = []
    traits_train = []

    X_val = []
    y_val = []
    traits_val = []

    X_test = []
    y_test = []
    traits_test = []

    files_used = 0

    for file_name in os.listdir(INTERNAL_STATES_DIR):
        split_name, rest = file_name.split("_", 1)
        if rest.startswith(model_name):
            files_used += 1
            print(f"Adding data of file {file_name} to data...")
            with open(os.path.join(INTERNAL_STATES_DIR, file_name), 'rb') as handle:
                file_json = pickle.load(handle)
                for passage_data in file_json:
                    for sentence_data in passage_data["sentence_data"]:
                        target = get_shd_prediction(passage_data["prompt"]["answerable"], sentence_data["pred"]) #sentence_data["target"]
                        if target is None:
                            continue

                        # 'layer_50_last_token', 'layer_100_last_token', 'activations_layer_50_last_token', 'activations_layer_100_last_token', 'probability', 'entropy' as keys
                        internal_states = sentence_data["internal_states"][internal_states_name]
                        
                        trait = {
                            "quantization": passage_data["quantization"],
                            "answerable": passage_data["prompt"]["answerable"],
                            "chunk_size": passage_data["prompt"]["chunk_size"],
                            "chunks_per_prompt": passage_data["prompt"]["chunks_per_prompt"],
                            "prompt_template_name": passage_data["prompt"]["prompt_template_name"],
                            "target": target
                        }

                        if True: # split_name == "train" or split_name == "val":
                            if trait["answerable"] == True:
                                continue

                        if split_name == "train":
                            X_train.append(internal_states)
                            y_train.append(target)
                            traits_train.append(trait)
                        elif split_name == "test":
                            X_test.append(internal_states)
                            y_test.append(target)
                            traits_test.append(trait)
                        elif split_name == "val":
                            X_val.append(internal_states)
                            y_val.append(target)
                            traits_val.append(trait)

    X_train = np.array(X_train)
    y_train = np.array(y_train, dtype=bool)

    X_val = np.array(X_val)
    y_val = np.array(y_val, dtype=bool)

    X_test = np.array(X_test)
    y_test = np.array(y_test, dtype=bool)
    
    cols = ["target"] # "answerable", "chunk_size", "chunks_per_prompt", "prompt_template_name", 
    # if files_used > 3:
    #     cols.append("quantization")

    columns = ["quantization", "answerable", "chunk_size", "chunks_per_prompt", "prompt_template_name", "target"]
    
    if correct_imbalance_train:
        print("Balancing train set")
        pprint(data_disbtr(traits_train))
        if oversample:
            train_err, X_train, y_train, traits_train = oversample_responses(traits_train, X_train, y_train, columns, max_iter=10000, no_improv_of=0.005, no_improv_after=100, max_err=0.0001)
        else:
            train_err, X_train, y_train, traits_train = undersample_responses(traits_train, X_train, y_train, columns, max_iter=10000, no_improv_of=0.005, no_improv_after=100, max_err=0.0001)
        pprint(data_disbtr(traits_train))
        print("Error:", train_err)
    if correct_imbalance_test:
        print("Balancing test set")
        pprint(data_disbtr(traits_test))
        if oversample:
            test_err, X_test, y_test, traits_test = oversample_responses(traits_test, X_test, y_test, columns, max_iter=10000, no_improv_of=0.005, no_improv_after=100, max_err=0.0001)
        else:
            test_err, X_test, y_test, traits_test = undersample_responses(traits_test, X_test, y_test, columns, max_iter=10000, no_improv_of=0.005, no_improv_after=100, max_err=0.0001)
        pprint(data_disbtr(traits_test))
        print("Error:", test_err)
    if correct_imbalance_val:
        print("Balancing val set")
        pprint(data_disbtr(traits_val))
        if oversample:
            val_err, X_val, y_val, traits_val = oversample_responses(traits_val, X_val, y_val, columns, max_iter=10000, no_improv_of=0.005, no_improv_after=100, max_err=0.0001)
        else:
            val_err, X_val, y_val, traits_val = undersample_responses(traits_val, X_val, y_val, columns, max_iter=10000, no_improv_of=0.005, no_improv_after=100, max_err=0.0001)
        pprint(data_disbtr(traits_val))
        print("Error:", val_err)

    total_data_count = y_train.shape[0] + y_val.shape[0] + y_test.shape[0]
    print(f"Final train size: {round(y_train.shape[0] * 100 / total_data_count, 4)}%")
    print(f" Final test size: {round(y_test.shape[0] * 100 / total_data_count, 4)}%")
    print(f"  Final val size: {round(y_val.shape[0] * 100 / total_data_count, 4)}%")

    return X_train, X_val, X_test, y_train, y_val, y_test

def run(model_name, internal_states_name, runs=10, shuffle_y=False):
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(model_name, internal_states_name, correct_imbalance_train=True, correct_imbalance_val=True, correct_imbalance_test=True, oversample=True)

    if shuffle_y:
        np.random.shuffle(y_train)
        np.random.shuffle(y_val)
        np.random.shuffle(y_test)

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
            epochs=800,
            stop_when_not_improved_after=30
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

    for model_name, model_name_starts_dict in MODEL_NAME_STARTS.items():
        quant_results = {}
        for quant_name, model_name_start in model_name_starts_dict.items():
            internal_states_results = {
                "normal_target": {},
                "shuffled_target": {}
            }
            for internal_state_name in INTERNAL_STATE_NAMES:
                internal_states_results["normal_target"][internal_state_name]   = run(model_name_start, internal_state_name, shuffle_y=False)
                internal_states_results["shuffled_target"][internal_state_name] = run(model_name_start, internal_state_name, shuffle_y=True)

            quant_results[quant_name] = internal_states_results
        model_results[model_name] = quant_results

    with open(RESULTS_FILE, "w") as file:
        json.dump(model_results, file, default=str)