import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os

from hallu_clf import HallucinationClassifier, train_model, test_model, load_checkpoint, print_confusion_matrix, save_confusion_matrix_plot

NAME = "random"
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_FILE = os.path.join(MODELS_DIR, f"{NAME}_checkpoint.pth")
ROC_CURVE_FILE = os.path.join(MODELS_DIR, f"{NAME}_roc_curve_test.png")
CONF_MATRIX_FILE = os.path.join(MODELS_DIR, f"{NAME}_conf_matrix_text.png")

# TODO: temp random data
np.random.seed(0)
num_samples = 1000
num_features = 10
X = np.random.randn(num_samples, num_features)
y = np.random.randint(2, size=num_samples)

# Split the data into training, validation, and test sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

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
    epochs=10,
    stop_when_not_improved_after=3
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