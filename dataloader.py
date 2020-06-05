import glob
import numpy as np
import os
import random
from random import shuffle
import torch
from torch.utils.data import Dataset

from hparam import hparam as hp
from utils import mfccs_and_spec



class JVSDataset(Dataset):
    
    def __init__(self):

        if hp.training:
            self.path = hp.data.train_path_unprocessed
        else:
            self.path = hp.data.test_path_unprocessed
        self.speakers = glob.glob(os.path.dirname(self.path))
        shuffle(self.speakers)
        
    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        speaker = self.speakers[idx]
        wav_files = glob.glob(speaker+'/*.WAV')
        shuffle(wav_files)
        wav_files = wav_files[0:self.utterance_number]
        
        mel_dbs = []
        for f in wav_files:
            _, mel_db, _ = mfccs_and_spec(f, wav_process = True)
            mel_dbs.append(mel_db)
        return torch.Tensor(mel_dbs)



class JVSDatasetPreprocessed(Dataset):
    
    def __init__(self, spekers_dict, shuffle=True):
        # data path
        self.path = hp.data.train_path if hp.training else hp.data.test_path
        self.index2sp = {index: speaker for speaker, index in spekers_dict.items()}
        self.shuffle = shuffle

    def __len__(self):
        return len(self.index2sp)

    def __getitem__(self, idx):
        sizes = []
        for i, file in enumerate(glob.glob('./data/*/*/*.npy')):
            if i == 2152:
                print("")
                print("")
            arr = np.load(file)
            size = arr.shape[0]
            sizes.append(size)
        max_size = max(sizes)
        pass
        if self.shuffle:
            selected_index, = random.sample(self.index2sp.keys(), 1)  # select random speaker
            selected_speaker = self.index2sp[selected_index]
        else:
            selected_speaker = self.index2sp[idx]

        utters_list = []
        search_path = os.path.join(self.path, selected_speaker, '*.npy')
        for utter in glob.glob(search_path):
            utterance = np.load(utter)  # load utterance spectrogram of selected speaker
            flames = utterance.shape[0]

            # reshape to [n_flames, n_input_feat(5*39=195)]
            input_size = hp.train.num_input_size
            mel_dim = hp.train.num_mel_dim
            utterance = utterance[:(flames // input_size) * input_size, :]
            utterance = utterance.reshape(-1, input_size * mel_dim)
            utterance = torch.tensor(utterance)
            utters_list.append(utterance)

        return utters_list, selected_speaker
