import itertools

import torch
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

        for name, param in self.layers.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.tanh(x)
        out = x
        return out


class SimMatrixLoss(nn.Module):
    def __init__(self, device, sim_csv_path, gamma=1.0):
        super().__init__()
        self.device = device
        self.simmat = self._get_normalized_simmatrix(sim_csv_path)
        self.gamma = gamma

    def _get_normalized_simmatrix(self, path):
        simmat = pd.read_csv(path)
        return simmat

    def _gaussian_kernel(self, di, dj):
        return torch.exp(-self.gamma * torch.norm(di - dj, 2))

    def _get_partial_simmatrix(self, speakers):
        ret = self.simmat[speakers][speakers]
        return torch.tensor(ret).float().to(self.device)

    def _loss_simmat_re(self, D, S):
        W = torch.where(S > 0, torch.ones_like(S), torch.zeros_like(S))



    def forward(self, d_vectors, speakers):
        for vecs, speaker in zip(d_vectors, speakers):
            Ns = len(speaker)
            speaker_iter = range(Ns)
            gram_matrix = torch.zeros(Ns, Ns)
            for i, j in itertools.product(speaker_iter, speaker_iter):
                di, dj = vecs[i], vecs[j]
                gram_matrix[i, j] = self._gaussian_kernel(di, dj)
            partial_simmat = self._get_partial_simmatrix(speakers)
            loss = self._loss_simmat_re(gram_matrix, partial_simmat)
