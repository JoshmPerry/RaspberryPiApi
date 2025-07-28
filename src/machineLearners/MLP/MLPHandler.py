import utils.MLPHelper as helper
import machineLearners.MLP.MLP as mlp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


# Hyperparameters
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 40
BATCH_SIZE = 32

class MLPHandler:
    def __init__(self):
        self.model = None
    def train(self, data, labels):
        # Shuffle data and labels in unison before splitting
        idx = np.random.permutation(len(data))
        data = np.array(data)[idx]
        labels = np.array(labels)[idx]

        train_x, valid_x, train_y, valid_y = helper.train_valid_split(data, labels, 1000)
        train_X_all = torch.tensor(train_x, dtype=torch.float32)
        valid_X_all = torch.tensor(valid_x, dtype=torch.float32)
        train_y_all = torch.tensor(train_y, dtype=torch.long)
        valid_y_all = torch.tensor(valid_y, dtype=torch.long)

        # Use DataLoader for batching and shuffling
        train_dataset = TensorDataset(train_X_all, train_y_all)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        self.model = mlp.MLP(input_size=train_X_all.shape[1], output_size=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(EPOCHS):
            self.model.train()
            total_loss = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Evaluate on validation set
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(valid_X_all)
                predictions = torch.argmax(outputs, dim=1)
                val_acc  = (predictions == valid_y_all).float().mean().item()

            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}, Val Accuracy: {val_acc:.4f}")

            # Save the best validation model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict()
        self.model.load_state_dict(best_model_state)

    def predict(self, data):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32)
            outputs = self.model(data_tensor)
            probs = torch.softmax(outputs, 0)
            print(probs)
            confidences, predicted = torch.max(probs, 0)
            print(predicted)
            print(confidences)
            return predicted.numpy(), confidences.numpy()