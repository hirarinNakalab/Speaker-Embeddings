import glob
import os
import librosa
from pysptk import conversion
import pyworld as pw
import numpy as np
import pandas as pd
from hparam import hparam as hp



def get_speakers_dict():
    genders_dict = {}
    genders = "male female".split()
    for gender in genders:
        csv_path = hp.data.sim_csv_path.format(gender)
        df = pd.read_csv(csv_path, header=None, index_col=0)
        speakers_dict = {speaker: index
                         for index, speaker in enumerate(df.index)}
        genders_dict[gender] = speakers_dict
    return genders_dict

def sp2mc(sp, order=39, alpha=0.41):   # alpha is all-pass constant
    mcep = conversion.sp2mc(sp, order, alpha)
    return mcep

def get_para(data, fs):
    _fo, _time = pw.dio(data, fs)               # 基本周波数の抽出
    fo = pw.stonemask(data, _fo, _time, fs)     # 基本周波数の修正
    sp = pw.cheaptrick(data, fo, _time, fs)     # スペクトル包絡の抽出
    mcep = sp2mc(sp)
    return fo, mcep

def wav_to_mcep(audio_path):
    mceps = []
    for i, utter_path in enumerate(audio_path):
        wav, source_sr = librosa.load(utter_path, sr=None)
        # Resample the wav to 16kHz
        wav = librosa.resample(wav, source_sr, hp.data.sr)
        wav = wav.astype(np.float)

        fo, mcep = get_para(wav, fs=hp.data.sr)
        if mcep.shape[0] < 50:
            print("remove {} because of less than 50 frames.".format(utter_path))
        # remove silence using fo info
        mask = fo.astype(np.bool)
        mcep = mcep[mask, 1:]

        mceps.append(mcep)
    return mceps



def main():
    print("start text independent utterance feature extraction...\n")
    os.makedirs(hp.data.train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(hp.data.test_path, exist_ok=True)    # make folder to save test file

    genders_dict = get_speakers_dict()
    n_totals = {gender: len(genders_dict[gender])
               for gender in genders_dict.keys()}
    n_trains = {gender: (n_totals[gender]//10)*9
               for gender in genders_dict.keys()} # split total data 90% train and 10% test

    for gender in genders_dict.keys():
        print("total {} speaker number : {}".format(gender, n_totals[gender]))
        print("train : {}, test : {}\n".format(
            n_trains[gender], n_totals[gender] - n_trains[gender]))

    for gender in genders_dict.keys():
        speakers = genders_dict[gender]
        n_train = n_trains[gender]
        for i, speaker in enumerate(speakers.keys()):
            print("%dth speaker processing..." % (i + 1))
            speaker_path = os.path.join(hp.unprocessed_data,
                                        speaker, hp.data.contents, '*.wav')
            audio_path = glob.glob(speaker_path)
            mceps = wav_to_mcep(audio_path)

            # save mcep as numpy file
            save_dir = os.path.join("{}", speaker)
            save_path = os.path.join(save_dir, "utter{:02d}.npy")
            if i < n_train:
                os.makedirs(save_dir.format(hp.data.train_path), exist_ok=True)
                for i, mcep in enumerate(mceps, start=1):
                    np.save(save_path.format(hp.data.train_path, i), mcep)
            else:
                os.makedirs(save_dir.format(hp.data.test_path), exist_ok=True)
                for i, mcep in enumerate(mceps, start=1):
                    np.save(save_path.format(hp.data.test_path, i), mcep)



if __name__ == "__main__":
    main()