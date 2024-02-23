import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

class DeepNN(torch.nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = torch.nn.Linear(1, 64)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 32)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(32, 4)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(len(class_names), len(class_names)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def preprocess_input(value):
    return torch.tensor(value, dtype=torch.float32).view(-1, 1)

def load_model(model_path):
    model = DeepNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def classify_input(model, input_value):
    with torch.no_grad():
        model.eval()
        input_tensor = preprocess_input(input_value)
        output = model(input_tensor)
        predicted_class = torch.argmax(output).item()
    return predicted_class

def generate_histogram(data):
    plt.hist(data, bins=20, color='blue', alpha=0.7)
    plt.title('Histogram of Data')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.show()

def generate_scatter_plot(x, y, xlabel, ylabel):
    plt.scatter(x, y, color='red', alpha=0.5)
    plt.title('Scatter Plot')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_value>")
        sys.exit(1)

    model_path = 'best_model.pth'
    model = load_model(model_path)

    # Load true labels from CSV file
    csv_file = 'data.csv'
    column_names = ['value', 'class']
    df = pd.read_csv(csv_file, header=None, names=column_names)

    # Extract inputs and true labels
    inputs = torch.tensor(df['value'].values, dtype=torch.float32).view(-1, 1)
    true_labels = torch.tensor(df['class'].values, dtype=torch.long)

    input_value = float(sys.argv[1])
    predicted_class = classify_input(model, input_value)

    print(f'The predicted class for input value {input_value} is: {predicted_class}')

    # Generate confusion matrix
    class_names = np.unique(true_labels.numpy()).astype(str)
    plot_confusion_matrix(true_labels.numpy(), np.array([predicted_class] * len(true_labels)), class_names)

    # Generate histogram
    generate_histogram(df['value'].values)

    # Generate scatter plot (assuming 'value' is on x-axis and 'class' is on y-axis)
    generate_scatter_plot(df['value'].values, df['class'].values, 'Value', 'Class')

if __name__ == '__main__':
    main()