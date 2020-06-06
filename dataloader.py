import glob
import numpy as np
import os
import random
from random import shuffle
import torch
from torch.utils.data import Dataset

from hparam import hparam as hp



def get_max_flames():
    sizes = []
    for i, file in enumerate(glob.glob('./data/*/*/*.npy')):
        arr = np.load(file)
        size = arr.shape[0]
        sizes.append(size)
    max_size = max(sizes)
    return max_size

def standardization(x, axis=None, ddof=0):
    x_mean = x.mean(axis=axis, keepdims=True)
    x_std = x.std(axis=axis, keepdims=True, ddof=ddof)
    return (x - x_mean) / x_std



class JVSDataset(Dataset):
    
    def __init__(self, spekers_dict, device, model):
        # data path
        self.path = hp.data.train_path if hp.training else hp.data.test_path
        self.index2sp = {index: speaker
                         for speaker, index in spekers_dict.items()
                         if index < (len(spekers_dict)//10)*9}
        self.input_size = hp.train.num_input_size
        self.mel_dim = hp.train.num_mel_dim
        self.shuffle = shuffle
        self.device = device
        self.model = model

    def __len__(self):
        return len(self.index2sp)

    def _get_dvector(self, utterance):
        outputs = self.model(utterance)
        return torch.mean(outputs, dim=0, keepdim=False)

    def _mel_to_tensor(self, mel):
        flames = mel.shape[0]
        # reshape to [n_flames, n_input_feat(5*39=195)]
        mel = mel[:(flames // self.input_size) * self.input_size, :]
        mel = mel.reshape(-1, self.input_size * self.mel_dim)
        mel = standardization(mel, axis=1)
        return torch.tensor(mel).float().to(self.device)

    def __getitem__(self, idx):
        selected_speaker = self.index2sp[idx]

        d_vectors = []
        search_path = os.path.join(self.path, selected_speaker, '*.npy')
        # select random utterances
        utters = random.sample(glob.glob(search_path), hp.train.M)
        for utter in utters:
            # load utterance spectrogram of selected speaker
            utterance = np.load(utter)
            mel = self._mel_to_tensor(utterance)
            d_vector = self._get_dvector(mel)
            d_vectors.append(d_vector)

        return d_vectors, selected_speaker
