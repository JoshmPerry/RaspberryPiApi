import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=60, hidden1=64, hidden2=32, hidden3=16, output_size=10):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, output_size)
        )

    def forward(self, x):
        return self.model(x)
