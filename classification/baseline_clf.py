import torch
import torch.nn as nn

class HallucinationClassifier(nn.Module):
    def __init__(self, input_size):
        super(HallucinationClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 8192)
        self.fc2 = nn.Linear(8192, 4096)
        self.fc3 = nn.Linear(4096, 1024)
        self.fc4 = nn.Linear(1024, 64)
        self.fc5 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, verbose: bool=True):
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

        # TODO: save best checkpoint and stop if not improved after x epochs
        val_acc, val_loss = validate_model(model, val_loader, criterion, verbose)

def validate_model(model, val_loader, criterion, verbose: bool=True):
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

def test_model(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            outputs = model(inputs)
            predicted = torch.round(outputs)
            predictions.extend(predicted.squeeze().tolist())
    return predictions
