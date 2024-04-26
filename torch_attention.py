from classifiers.pytorchAttention import AttentionModel
import torch

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class RandomDataset(Dataset):
    def __init__(self, num_samples, input_size, num_classes):
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes

        # Generate random input data
        self.data = torch.randn(num_samples, input_size)

        # Generate random labels
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Define hyperparameters
emb_size = 64
num_heads = 4
dropout = 0.1
num_classes = 10
dim_ff = 128
learning_rate = 0.001
batch_size = 32
num_epochs = 10
num_samples = 1000  # Number of samples in the dataset
input_size = 300  # Size of input data
num_classes = 10  # Number of classes

# Create random dataset
random_dataset = RandomDataset(num_samples, input_size, num_classes)

# Create data loader
train_loader = DataLoader(random_dataset, batch_size=batch_size, shuffle=True)

# Define the model
model = AttentionModel(emb_size, num_heads, dropout, num_classes, dim_ff)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    epoch_loss = total_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')