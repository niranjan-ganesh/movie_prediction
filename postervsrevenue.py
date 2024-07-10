import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load CSV file
csv_file = 'id_popularity.csv'
data_df = pd.read_csv(csv_file)

# Ensure 'revenue' is numeric
data_df['revenue'] = pd.to_numeric(data_df['revenue'], errors='coerce')

# Normalize 'revenue' between 1 and 10
min_revenue = data_df['revenue'].min()
max_revenue = data_df['revenue'].max()
data_df['revenue'] = 1 + (data_df['revenue'] - min_revenue) * 9 / (max_revenue - min_revenue)

class MovieDataset(Dataset):
    def __init__(self, data_df, root_dir, transform=None):
        self.data_df = data_df
        self.root_dir = root_dir
        self.transform = transform
        # Filter rows where 'download_successful' is 'Yes'
        self.data_df = self.data_df[self.data_df['download_successful'] == 'Yes']

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        movie_id = self.data_df.iloc[idx]["id"]
        score = self.data_df.iloc[idx]["revenue"]
        img_path = os.path.join(self.root_dir, str(movie_id), f'{movie_id}.jpg')
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Convert score to float and then to tensor
        score = float(score)

        return image, torch.tensor(score, dtype=torch.float32)

# Transforming the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Initialize the dataset
root_dir = '/Users/niranjanganesan/PycharmProjects/pythonProject12/posters'
dataset = MovieDataset(data_df=data_df, root_dir=root_dir, transform=transform)

# Split dataset into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Initialize DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class ShallowCNN(nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 64 * 64)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ShallowCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs=20, patience=5):
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(torch.float32)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Patience: {patience_counter}')

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return loss_history

# Train the model with dynamic early stopping
loss_history = train_model(model, train_loader, criterion, optimizer, epochs=50, patience=5)

# Plot training loss
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(torch.float32)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()

            # Calculate accuracy for this batch
            predictions = outputs.squeeze().numpy()
            labels = labels.numpy()
            correct_predictions += np.sum(np.abs(predictions - labels) < 0.1)
            total_predictions += labels.size

    average_loss = test_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    print(f'Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')
    return average_loss, accuracy

# Evaluate the model on the test set
evaluate_model(model, test_loader, criterion)
