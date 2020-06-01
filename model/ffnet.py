import torch.nn as nn
import torch.nn.functional as F



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


class FrobeniusNorm(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass