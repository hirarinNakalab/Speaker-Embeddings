import glob
import numpy as np
import os
import random
from random import shuffle
import torch
from torch.utils.data import Dataset

from hparam import hparam as hp
from utils import mfccs_and_spec



class SpeakerDatasetJVS(Dataset):
    
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



class SpeakerDatasetJVSPreprocessed(Dataset):
    
    def __init__(self, spekers_list, shuffle=True):
        # data path
        self.path = hp.data.train_path if hp.training else hp.data.test_path
        self.spekers_list = spekers_list
        self.shuffle = shuffle

    def __len__(self):
        return len(self.spekers_list)

    def __getitem__(self, idx):
        if self.shuffle:
            selected_speaker = random.sample(self.spekers_list, 1)[0]  # select random speaker
        else:
            selected_speaker = self.spekers_list[idx]

        utters_list = []
        for utter in os.listdir(os.path.join(self.path, selected_speaker)):
            utterance = np.load(utter)  # load utterance spectrogram of selected speaker
            flames = utterance.shape[0]
            # reshape to [n_flames, n_input_feat(5*39=195)]
            utterance = utterance[:(flames // hp.num_input_flames) * hp.num_input_flames, :]
            utterance = utterance.reshape(-1, hp.num_input_flames * hp.num_mel_dimension)
            utterance = torch.tensor(utterance)
            utters_list.append(utterance)

        return utters_list, selected_speaker
