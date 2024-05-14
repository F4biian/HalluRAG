import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pickle
from pprint import pprint

from hallu_clf import HallucinationClassifier, train_model, test_model, load_checkpoint, print_confusion_matrix, save_confusion_matrix_plot

MODEL_NAME = "meta-llama_Llama-2-7b-chat-hf"
INTERNAL_STATE_NAME = "layer_50_last_token" # 'layer_50_last_token', 'layer_100_last_token', 'activations_layer_50_last_token', 'activations_layer_100_last_token', 'probability', 'entropy' as keys

NAME = f"{MODEL_NAME}+{INTERNAL_STATE_NAME}"
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, "models")
DATA_DIR = os.path.join(os.path.join(CURR_DIR, ".."), "data")
INTERNAL_STATES_DIR = os.path.join(os.path.join(DATA_DIR, "RAGTruth"), "internal_states")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_FILE = os.path.join(MODELS_DIR, f"{NAME}_checkpoint.pth")
ROC_CURVE_FILE = os.path.join(MODELS_DIR, f"{NAME}_roc_curve_test.png")
CONF_MATRIX_FILE = os.path.join(MODELS_DIR, f"{NAME}_conf_matrix_text.png")

np.random.seed(432)

X = []
y = []
passage_ids = []

for file_name in os.listdir(INTERNAL_STATES_DIR):
    if file_name.startswith(MODEL_NAME):
        print(f"Adding data of file {file_name} to data...")
        with open(os.path.join(INTERNAL_STATES_DIR, file_name), 'rb') as handle:
            file_json = pickle.load(handle)
            for passage_data in file_json:
                print(passage_data.keys())
                for sentence_data in passage_data["sentence_data"]:
                    target = sentence_data["target"]

                    # 'layer_50_last_token', 'layer_100_last_token', 'activations_layer_50_last_token', 'activations_layer_100_last_token', 'probability', 'entropy' as keys
                    internal_states = sentence_data["internal_states"][INTERNAL_STATE_NAME]
                    
                    X.append(internal_states)
                    y.append(target)
                    exit()
                    passage_ids.append(passage_data[""])

X = np.array(X)
y = np.array(y, dtype=bool)

# TODO: data leak in two three ways possibile:!!!!!!!!
# 1) oversampling before train_test_split
# 2) same sentence is available in various quantizations
# 3) there are multiple sentences per passage (so states of first 2 cum. sentences is in test but first 3 cum. sentences were in train)

print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")
print(f"y.mean(): {y.mean()}")

def correct_binary_imbalance(X, y, oversampling=False):
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

    return X[all_indices], y[all_indices]

print("Rebalancing data based on target...")
X, y = correct_binary_imbalance(X, y)

print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")
print(f"y.mean(): {y.mean()}")

# # Random data
# num_samples = 1000
# num_features = 10
# X = np.random.randn(num_samples, num_features)
# y = np.random.randint(2, size=num_samples)

# Split the data into training, validation, and test sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.3, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=0)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

print(f"X_train.size(): {X_train_tensor.size()}")
print(f"y_train.size(): {y_train_tensor.size()}")
print(f"X_val_tensor.size(): {X_val_tensor.size()}")
print(f"y_val_tensor.size(): {y_val_tensor.size()}")
print(f"X_test_tensor.size(): {X_test_tensor.size()}")
print(f"y_test_tensor.size(): {y_test_tensor.size()}")

print(f"y_train_tensor.mean(): {y_train_tensor.mean()}")
print(f"y_val_tensor.mean(): {y_val_tensor.mean()}")
print(f"y_test_tensor.mean(): {y_test_tensor.mean()}")

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

# Defining model, loss and optimizer
model = HallucinationClassifier(X_train.shape[1]).to(DEVICE)
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

test_loss, test_acc, test_precision, test_recall, test_auc, test_conf_matrix = test_model(model, test_loader, criterion, ROC_CURVE_FILE)
print("-"*30)
print(f"Test Loss:\t{test_loss}")
print(f"Test Accuracy:\t{test_acc}")
print(f"Test Precision:\t{test_precision}")
print(f"Test Recall:\t{test_recall}")
print(f"Test AUC:\t{test_auc}")
print("-"*30)
print_confusion_matrix(test_conf_matrix)
print("-"*30)
save_confusion_matrix_plot(test_conf_matrix, CONF_MATRIX_FILE)

# load_checkpoint(CHECKPOINT_FILE, model, optimizer)