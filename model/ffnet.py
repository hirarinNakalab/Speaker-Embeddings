import itertools
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
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
        self.max_similarity = 1.0
        self.scaler = MinMaxScaler()
        self.simmat = self._get_normalized_simmatrix(sim_csv_path)
        self.gamma = gamma

    def _get_normalized_simmatrix(self, path):
        simmat = pd.read_csv(path, header=None, index_col=0)
        self.sp2idx = {speaker: index
                       for index, speaker in enumerate(simmat.index)}
        simmat = self.scaler.fit_transform(simmat)
        return simmat

    def _gaussian_kernel(self, di, dj):
        return torch.exp(-self.gamma * torch.norm(di - dj, 2))

    def _get_partial_simmatrix(self, speakers):
        Ns = len(speakers)
        ret = torch.zeros(Ns, Ns)
        sp_ids = [self.sp2idx[sp] for sp in speakers]
        ret_ids = {sp_id: i for i, sp_id in enumerate(sp_ids)}
        for i, j in itertools.product(sp_ids, sp_ids):
            row, col = ret_ids[i], ret_ids[j]
            val = self.simmat[i, j]
            ret[row, col] = val
        return torch.tensor(ret).float().to(self.device)

    def _loss_simmat_re(self, Kd, S):
        W = (S > 0).float().to(self.device)
        Is = torch.eye(W.size(0)).to(self.device)
        denom = torch.norm(W - Is, p="fro")

        Kd_tilde = Kd - torch.mul(Kd, Is)
        S_tilde = S - self.max_similarity * Is
        diff = Kd_tilde - S_tilde
        diff_normed = torch.mul(W, diff)
        numer = torch.norm(diff_normed, p="fro")

        loss = 2 * (numer / denom)
        return loss


    def forward(self, d_vectors, speakers):
        Ns = len(speakers)
        speaker_iter = range(Ns)

        losses = []
        for vecs in d_vectors:
            gram_matrix = torch.zeros(Ns, Ns).to(self.device)
            for i, j in itertools.product(speaker_iter, speaker_iter):
                di, dj = vecs[i], vecs[j]
                gram_matrix[i, j] = self._gaussian_kernel(di, dj)
            partial_simmat = self._get_partial_simmatrix(speakers)
            loss = self._loss_simmat_re(gram_matrix, partial_simmat)
            losses.append(loss)
        return torch.sum(torch.stack(losses))