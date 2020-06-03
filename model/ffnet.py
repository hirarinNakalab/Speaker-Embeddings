import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class FFNet(nn.Module):

    def __init__(self, input_size=5*39, hidden_size=256, d_vector_size=8):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bottle_neck = nn.Linear(hidden_size, d_vector_size)

        self.layers = nn.ModuleList(
            [self.fc1, self.fc2, self.fc3, self.bottle_neck])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.tanh(x)
        out = x
        return out


class SimMatrixLoss(nn.Module):
    def __init__(self, sim_csv_path):
        super().__init__()
        self.simmat = self.get_normalized_simmatrix(sim_csv_path)
        pass

    def get_normalized_simmatrix(self, path):
        simmat = pd.read_csv(path)
        return simmat

    def forward(self, x):
        pass