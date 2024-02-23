import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def plot_loss_accuracy(train_loss_list, val_loss_list, train_acc_list, val_acc_list, mean_error_train, mean_error_val, mean_error_test):
    # Plot Training Loss and Accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='Training Accuracy', color='orange')
    plt.plot(val_acc_list, label='Validation Accuracy', color='red')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f'Mean Error - Training: {mean_error_train:.2f}%')
    # Plot Validation Loss and Accuracy
    print(f'Mean Error - Validation: {mean_error_val:.2f}%, Test: {mean_error_test:.2f}%')

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def calculate_mean_error(model, criterion, data_loader, device):
    model.eval()
    total_error = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_error += loss.item()

    mean_error = total_error / len(data_loader)
    return mean_error * 100

def train(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    predictions_list = []
    labels_list = []
    outputs_list = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        predictions_list.extend(predictions.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())
        outputs_list.append((outputs,))

    average_loss = running_loss / len(train_loader)
    accuracy = accuracy_score(labels_list, predictions_list) * 100

    return average_loss, accuracy, outputs_list

def validate(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    predictions_list = []
    labels_list = []
    outputs_list = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            predictions_list.extend(predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            outputs_list.append((outputs,))

    average_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(labels_list, predictions_list) * 100

    return average_loss, accuracy, outputs_list

def main():
    num_epochs = 5000  # You can adjust this value as needed

    csv_file = 'data.csv'  
    column_names = ['value', 'class'] 

    df = pd.read_csv(csv_file, header=None, names=column_names)

    inputs = torch.tensor(df['value'].values, dtype=torch.float32).view(-1, 1)
    labels = torch.tensor(df['class'].values, dtype=torch.long)

    # Split the dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(inputs, labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create DataLoader for training set
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create DataLoader for validation set
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create DataLoader for test set
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = DeepNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_accuracy = 0.0
    best_model_state = None

    interval = 30
    loss_list = []
    accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    val_outputs_list = []

    for epoch in range(num_epochs):
        train_loss, train_accuracy, _ = train(model, criterion, optimizer, train_loader, device)
        val_loss, val_accuracy, val_outputs = validate(model, criterion, val_loader, device)

        
        if (epoch + 1) % interval == 0:
            loss_list.append(train_loss)
            accuracy_list.append(train_accuracy)
            val_loss_list.append(val_loss)
            val_accuracy_list.append(val_accuracy)
            val_outputs_list.extend(val_outputs)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_state = model.state_dict()
                print('Best model updated. Saving...')

    torch.save(best_model_state, 'best_model.pth')

    mean_error_train = calculate_mean_error(model, criterion, train_loader, device)
    mean_error_val = calculate_mean_error(model, criterion, val_loader, device)
    mean_error_test = calculate_mean_error(model, criterion, test_loader, device)

    plot_loss_accuracy(loss_list, val_loss_list, accuracy_list, val_accuracy_list, mean_error_train, mean_error_val, mean_error_test)

    model.eval()
    with torch.no_grad():
        val_outputs = torch.cat([outputs[0] for outputs in val_outputs_list], dim=0)
        val_predictions = torch.argmax(val_outputs, dim=1)
        val_acc = accuracy_score(y_val.cpu().numpy(), val_predictions.cpu().numpy()) * 100

    plot_confusion_matrix(y_val.cpu().numpy(), val_predictions.cpu().numpy(), classes=df['class'].unique())

if __name__ == '__main__':
    main()