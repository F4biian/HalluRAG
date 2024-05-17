import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class HallucinationClassifier(nn.Module):
    def __init__(self, input_size) -> None:
        super(HallucinationClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        checkpoint_file: str,
        epochs=10,
        stop_when_not_improved_after: int=5,
        verbose: bool=True
    ) -> None:
    best_epoch = None
    not_improved_streak = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader.dataset)}")

        val_acc, val_loss = validate_model(model, val_loader, criterion, verbose)

        if best_epoch is None or best_epoch["val_acc"] < val_acc:
            if verbose:
                print(f"Model improved val_acc from {best_epoch['val_acc'] if best_epoch else 'None'} to {val_acc}!")
            best_epoch = {
                "val_acc": val_acc,
                "val_loss": val_loss,
                "loss": running_loss
            }
            save_checkpoint(model, optimizer, epoch, checkpoint_file, verbose)
            not_improved_streak = 0
        else:
            not_improved_streak += 1

        if not_improved_streak >= stop_when_not_improved_after:
            if verbose:
                print(f"Model has not improved since {not_improved_streak} epochs! Stopped training.")
            break
    
    if verbose:
        print("Done!")

def validate_model(model: nn.Module, val_loader: DataLoader, criterion: nn.modules.loss._Loss, verbose: bool=True) -> Tuple[float, float]:
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            total_loss += loss.item() * inputs.size(0)
            predicted = torch.round(outputs)
            total_correct += (predicted == labels.unsqueeze(1)).sum().item()
            total_samples += labels.size(0)
    if verbose:
        print(f"Validation Loss: {total_loss / len(val_loader.dataset)}, Accuracy: {(total_correct / total_samples) * 100}%")
    return total_correct / total_samples, total_loss

def test_model(model: nn.Module, test_loader: DataLoader, criterion: nn.modules.loss._Loss, roc_curve_file: str) -> Tuple[float, float, float, float, float, np.ndarray]:
    model.eval()
    predictions = []
    predictions_probabilities = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            total_loss += loss.item() * inputs.size(0)
            predictions_probabilities.extend(outputs.squeeze().tolist())
            predicted = torch.round(outputs).squeeze().tolist()
            predictions.extend(predicted)
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, predictions)
    confusion = confusion_matrix(all_labels, predictions)

    true_positive = confusion[1, 1]
    false_positive = confusion[0, 1]
    false_negative = confusion[1, 0]
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    fpr, tpr, _ = roc_curve(all_labels, predictions_probabilities)
    roc_auc = auc(fpr, tpr)

    save_roc_curve_plot(fpr, tpr, roc_auc, roc_curve_file)

    f1 = (2 * precision * recall) / (precision + recall)

    return total_loss / len(test_loader.dataset), accuracy, precision, recall, f1, roc_auc, confusion

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, filepath: str, verbose: bool=True) -> None:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)

    if verbose:
        print(f"Checkpoint saved at {filepath}")
    
def load_checkpoint(filepath: str, model: nn.Module, optimizer: torch.optim.Optimizer=None, verbose: bool=True) -> None:
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    if verbose:
        print(f"Checkpoint loaded from {filepath}, starting from epoch {epoch}")

    model.eval()

def print_confusion_matrix(conf_matrix: np.ndarray) -> None:
    print("Confusion Matrix:")
    for row in conf_matrix:
        print("\t".join(map(str, row)))

def save_confusion_matrix_plot(conf_matrix: np.ndarray, filepath: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(filepath)
    plt.close()

def save_roc_curve_plot(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, filepath: str) -> None:
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(filepath)
    plt.close()