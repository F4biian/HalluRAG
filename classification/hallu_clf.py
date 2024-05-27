import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Tuple, Dict, Any
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, recall_score, precision_score, f1_score, cohen_kappa_score, matthews_corrcoef
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class HallucinationClassifier(nn.Module):
    def __init__(self, input_size, dropout_p=0.15) -> None:
        super(HallucinationClassifier, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
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
        print()
        model.train()
        total_correct = 0
        total_samples = 0
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            predicted = torch.round(outputs)
            running_loss += loss.item() * inputs.size(0)
            total_correct += (predicted == labels.unsqueeze(1)).sum().item()
            total_samples += labels.size(0)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, loss: {round(running_loss / len(train_loader.dataset), 4)}, acc: {round((total_correct / total_samples) * 100, 4)}%")

        val_acc, val_loss = validate_model(model, val_loader, criterion, verbose)

        if best_epoch is None or best_epoch["val_loss"] > val_loss:
            if verbose:
                print(f"Model improved val_loss from {best_epoch['val_loss'] if best_epoch else 'None'} to {val_loss}!")
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
        print(f"val_loss: {round(total_loss / len(val_loader.dataset), 4)}, val_acc: {round((total_correct / total_samples) * 100, 4)}%")
    return total_correct / total_samples, total_loss / len(val_loader.dataset)

def test_model(model: nn.Module, test_loader: DataLoader, criterion: nn.modules.loss._Loss, thresholds: dict={}) -> Dict[str, Any]:
    results = {}

    model.eval()
    predictions_probabilities = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            total_loss += loss.item() * inputs.size(0)

            squeezed_output = outputs.squeeze().tolist()
            if type(squeezed_output) == float:
                squeezed_output = [squeezed_output]
            predictions_probabilities.extend(squeezed_output)

            all_labels.extend(labels.tolist())

    results["loss"] = total_loss / len(test_loader.dataset)

    results = {
        **results,
        **get_results(all_labels, predictions_probabilities, **thresholds)
    }

    return results

def get_best_threshold(all_labels: list, predictions_probabilities: list, method: callable, bounds=(0, 1), num=101, maximize=True) -> float:
    scores = pd.Series(dtype=float)

    for th in np.linspace(bounds[0], bounds[1], num=num):
        y_pred = predictions_probabilities > th
        score = method(all_labels, y_pred)
        scores[th] = score
    
    best_threshold = scores.sort_values(ascending=maximize).index[-1]
    return best_threshold

def get_random_classifier_results(test_loader: DataLoader, threshold=0.5) -> Dict[str, Any]:
    predictions_probabilities = []
    all_labels = []
    for inputs, labels in test_loader:
        outputs = torch.FloatTensor(np.random.rand(inputs.size()[0]))
        predictions_probabilities.extend(outputs.tolist())
        all_labels.extend(labels.tolist())
    return get_results(all_labels, predictions_probabilities, threshold, threshold, threshold, threshold, threshold, threshold, threshold, threshold, threshold, threshold)

def get_results(all_labels: list, predictions_probabilities: list, mcc_thr=None, cohen_kappa_thr=None, accuracy_thr=None, normalized_accuracy_thr=None, recall_hallucinated_thr=None, precision_hallucinated_thr=None, f1_hallucinated_thr=None, recall_grounded_thr=None, precision_grounded_thr=None, f1_grounded_thr=None) -> Dict[str, Any]:
    predictions_probabilities = np.array(predictions_probabilities)
    results = {}

    if cohen_kappa_thr is None:
        cohen_kappa_thr = get_best_threshold(all_labels, predictions_probabilities, lambda y_true, y_pred: cohen_kappa_score(y_true, y_pred))
    results["cohen_kappa_threshold"] = cohen_kappa_thr
    results["cohen_kappa"] = cohen_kappa_score(all_labels, predictions_probabilities > cohen_kappa_thr)

    if mcc_thr is None:
        mcc_thr = get_best_threshold(all_labels, predictions_probabilities, lambda y_true, y_pred: matthews_corrcoef(y_true, y_pred))
    results["mcc_threshold"] = mcc_thr
    results["mcc"] = matthews_corrcoef(all_labels, predictions_probabilities > mcc_thr)

    if accuracy_thr is None:
        accuracy_thr = get_best_threshold(all_labels, predictions_probabilities, lambda y_true, y_pred: accuracy_score(y_true, y_pred, normalize=False))
    results["accuracy_threshold"] = accuracy_thr
    if normalized_accuracy_thr is None:
        normalized_accuracy_thr = get_best_threshold(all_labels, predictions_probabilities, lambda y_true, y_pred: accuracy_score(y_true, y_pred, normalize=True))
    results["normalized_accuracy_threshold"] = accuracy_thr

    results["accuracy"] = accuracy_score(all_labels, predictions_probabilities > accuracy_thr, normalize=False)
    results["normalized_accuracy"] = accuracy_score(all_labels, predictions_probabilities > normalized_accuracy_thr, normalize=True)
    results["confusion_matrix"] = confusion_matrix(all_labels, predictions_probabilities > accuracy_thr).tolist()

    # Metrics for hallucination detection
    if recall_hallucinated_thr is None:
        recall_hallucinated_thr = get_best_threshold(all_labels, predictions_probabilities, lambda y_true, y_pred: recall_score(y_true, y_pred, pos_label=1))
    results["recall_hallucinated_threshold"] = recall_hallucinated_thr
    if precision_hallucinated_thr is None:
        precision_hallucinated_thr = get_best_threshold(all_labels, predictions_probabilities, lambda y_true, y_pred: precision_score(y_true, y_pred, pos_label=1))
    results["precision_hallucinated_threshold"] = precision_hallucinated_thr
    if f1_hallucinated_thr is None:
        f1_hallucinated_thr = get_best_threshold(all_labels, predictions_probabilities, lambda y_true, y_pred: f1_score(y_true, y_pred, pos_label=1))
    results["f1_hallucinated_threshold"] = f1_hallucinated_thr

    results["recall_hallucinated"] = recall_score(all_labels, predictions_probabilities > recall_hallucinated_thr, pos_label=1)
    results["precision_hallucinated"] = precision_score(all_labels, predictions_probabilities > precision_hallucinated_thr, pos_label=1)
    results["f1_hallucinated"] = f1_score(all_labels, predictions_probabilities > f1_hallucinated_thr, pos_label=1)

    fpr_hallucinated, tpr_hallucinated, _ = roc_curve(all_labels, predictions_probabilities, pos_label=1)
    results["fpr_hallucinated"] = fpr_hallucinated.tolist()
    results["tpr_hallucinated"] = tpr_hallucinated.tolist()
    results["roc_auc_hallucinated"] = auc(fpr_hallucinated, tpr_hallucinated)

    P_hallucinated, R_hallucinated, _ = precision_recall_curve(all_labels, predictions_probabilities, pos_label=1)
    results["P_hallucinated"] = P_hallucinated.tolist()
    results["R_hallucinated"] = R_hallucinated.tolist()
    results["auc_pr_hallucinated"] = auc(R_hallucinated, P_hallucinated)

    # Metrics for grounded statements detection
    if recall_grounded_thr is None:
        recall_grounded_thr = get_best_threshold(all_labels, predictions_probabilities, lambda y_true, y_pred: recall_score(y_true, y_pred, pos_label=0))
    results["recall_grounded_threshold"] = recall_grounded_thr
    if precision_grounded_thr is None:
        precision_grounded_thr = get_best_threshold(all_labels, predictions_probabilities, lambda y_true, y_pred: precision_score(y_true, y_pred, pos_label=0))
    results["precision_grounded_threshold"] = precision_grounded_thr
    if f1_grounded_thr is None:
        f1_grounded_thr = get_best_threshold(all_labels, predictions_probabilities, lambda y_true, y_pred: f1_score(y_true, y_pred, pos_label=0))
    results["f1_grounded_threshold"] = f1_grounded_thr

    results["recall_grounded"] = recall_score(all_labels, predictions_probabilities > recall_grounded_thr, pos_label=0)
    results["precision_grounded"] = precision_score(all_labels, predictions_probabilities > precision_grounded_thr, pos_label=0)
    results["f1_grounded"] = f1_score(all_labels, predictions_probabilities > f1_grounded_thr, pos_label=0)

    fpr_grounded, tpr_grounded, _ = roc_curve(all_labels, predictions_probabilities, pos_label=0)
    results["fpr_grounded"] = fpr_grounded.tolist()
    results["tpr_grounded"] = tpr_grounded.tolist()
    results["roc_auc_grounded"] = auc(fpr_grounded, tpr_grounded)

    P_grounded, R_grounded, _ = precision_recall_curve(all_labels, predictions_probabilities, pos_label=0)
    results["P_grounded"] = P_grounded.tolist()
    results["R_grounded"] = R_grounded.tolist()
    results["auc_pr_grounded"] = auc(R_grounded, P_grounded)

    return results

def get_thresolds_from_results(results: dict) -> dict:
    thrs = {}
    for key, value in results.items():
        if key.endswith("threshold"):
            thrs[key.replace("threshold", "thr")] = value
    return thrs

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