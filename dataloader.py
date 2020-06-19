import glob
import numpy as np
import os
import random
import librosa
from random import shuffle
from numpy.lib.stride_tricks import as_strided

import torch
from torch.utils.data import Dataset

from hparam import hparam as hp
from preprocess import get_para



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

def mel_to_tensor(mel, device):
    n_flames = mel.shape[0]
    shape = (n_flames - hp.train.num_input_size + 1,
             hp.train.num_input_size,
             hp.train.num_mel_dim)

    strides = mel.strides
    stride_shape = (strides[0], strides[0], strides[1])

    strided = as_strided(mel, shape=shape, strides=stride_shape)
    strided = strided.reshape(shape[0], -1)

    mel = standardization(strided, axis=1)
    return torch.Tensor(mel).to(device)

def get_dvector(model, utterance):
    outputs = model(utterance)
    return torch.mean(outputs, dim=0, keepdim=False)

def audio_to_dvector(audio_path, model, device):
    wav, source_sr = librosa.load(audio_path, sr=None)
    # Resample the wav to 16kHz
    wav = librosa.resample(wav, source_sr, hp.data.sr)
    wav = wav.astype(np.float)

    fo, mcep = get_para(wav, fs=hp.data.sr)

    # remove silence using fo info
    mask = fo.astype(np.bool)
    mcep = mcep[mask, 1:]
    mel = mel_to_tensor(mcep, device)
    d_vector = get_dvector(model, mel)
    return d_vector


def utters_to_dvectors(utters, model, device):
    d_vectors = []
    for utter in utters:
        # load utterance spectrogram of selected speaker
        utterance = np.load(utter)
        mel = mel_to_tensor(utterance, device)
        d_vector = get_dvector(model, mel)
        d_vectors.append(d_vector)
    return d_vectors



class JVSNonparaVal(Dataset):

    def __init__(self, spekers_dict, device, model):
        # data path
        self.path = hp.data.nonpara_path
        self.index2sp = {index: speaker for speaker, index in spekers_dict.items()}
        self.input_size = hp.train.num_input_size
        self.mel_dim = hp.train.num_mel_dim
        self.shuffle = shuffle
        self.device = device
        self.model = model

    def __len__(self):
        return len(self.index2sp)

    def __getitem__(self, idx):
        selected_speaker = self.index2sp[idx]

        search_path = os.path.join(self.path, selected_speaker, '*.npy')
        # select random utterances
        utters = random.sample(glob.glob(search_path), hp.test.M)
        return utters_to_dvectors(utters, self.model, self.device)


class JVSNonparaTrain(Dataset):
    
    def __init__(self, spekers_dict, device, model):
        # data path
        self.path = hp.data.nonpara_path
        self.n_train_sp = (len(spekers_dict)//10)*9
        self.index2sp = {index: speaker
                         for speaker, index in spekers_dict.items()
                         if index < self.n_train_sp}
        self.input_size = hp.train.num_input_size
        self.mel_dim = hp.train.num_mel_dim
        self.shuffle = shuffle
        self.device = device
        self.model = model

    def __len__(self):
        return len(self.index2sp)

    def __getitem__(self, idx):
        selected_speaker = self.index2sp[idx]

        d_vectors = []
        search_path = os.path.join(self.path, selected_speaker, '*.npy')
        # select random utterances
        utters = random.sample(glob.glob(search_path), hp.train.M)
        return utters_to_dvectors(utters, self.model, self.device)