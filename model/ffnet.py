import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn



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
            x = torch.tanh(x)
        out = x
        return out


class SimMatrixLoss(nn.Module):
    def __init__(self, device, sim_csv_path, gamma=1.0):
        super().__init__()
        self.device = device
        self.max_similarity = 1.0
        self.scaler = MinMaxScaler()
        self.gamma = gamma

        self._get_normalized_simmatrix(sim_csv_path)

    def _get_normalized_simmatrix(self, path):
        simmat = pd.read_csv(path, header=None, index_col=0)
        self.sp2idx = {speaker: index
                       for index, speaker in enumerate(simmat.index)}

        W = np.array(simmat > 0).astype(np.float)
        self.W = torch.Tensor(W).to(self.device)
        simmat = self.scaler.fit_transform(simmat)
        self.simmat = torch.Tensor(simmat).to(self.device)

    def _gaussian_kernel(self, di, dj):
        diff = di - dj
        norm = torch.norm(diff, p=2, dim=1, keepdim=True)
        return torch.exp(-self.gamma * norm)

    def _loss_simmat_re(self, Kd, S, W):
        Ns = W.size(0)
        batch_size = Kd.shape[-1]

        S = S.unsqueeze(2).repeat(1, 1, batch_size)
        W = W.unsqueeze(2).repeat(1, 1, batch_size)
        Is = torch.eye(Ns).to(self.device).unsqueeze(2).repeat(1, 1, batch_size)

        denom = torch.norm(W - Is, p="fro")

        S_tilde = S - self.max_similarity * Is
        Kd_tilde = Kd - torch.mul(Kd, Is)
        diff = Kd_tilde - S_tilde
        diff_normed = torch.mul(W, diff)
        numer = torch.norm(diff_normed, p="fro")

        loss = 2 * (numer / denom)
        return loss


    def forward(self, d_vectors):
        Ns = d_vectors[0].shape[0]

        utter = torch.stack(d_vectors).permute(1, 2, 0)
        ks = [self._gaussian_kernel(utter, utter[i]) for i in range(Ns)]
        gram_matrix = torch.cat(ks, dim=1)

        part_sim, part_W = self.simmat[:Ns, :Ns], self.W[:Ns, :Ns]

        loss = self._loss_simmat_re(gram_matrix, part_sim, part_W)
        return loss