import glob
import os
import librosa
from pysptk import conversion
import pyworld as pw
import numpy as np
from hparam import hparam as hp



def get_speaker_list():
    males, females = [], []
    with open(hp.data.gender_data_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0: continue
            values = line.strip().split(" ")
            if values[1] == "M":
                males.append(values[0])
            else:
                females.append(values[0])
    return males, females

def sp2mc(sp, order=39, alpha=0.41):   # alpha is all-pass constant
    mcep = conversion.sp2mc(sp, order, alpha)
    return mcep

def get_para(data, fs):
    _fo, _time = pw.dio(data, fs)               # 基本周波数の抽出
    fo = pw.stonemask(data, _fo, _time, fs)     # 基本周波数の修正
    sp = pw.cheaptrick(data, fo, _time, fs)     # スペクトル包絡の抽出
    mcep = sp2mc(sp)
    return fo, mcep

def main():
    print("start text independent utterance feature extraction")
    os.makedirs(hp.data.train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(hp.data.test_path, exist_ok=True)    # make folder to save test file

    males, females = get_speaker_list()
    n_total_males = len(males)
    n_total_females = len(females)
    n_train_males = (n_total_males//10)*9            # split total data 90% train and 10% test
    n_train_females = (n_total_females//10)*9            # split total data 90% train and 10% test

    print("total male speaker number : %d" % n_total_males)
    print("train : %d, test : %d" % (n_train_males, n_total_males - n_train_males))
    print("total female speaker number : %d" % n_total_females)
    print("train : %d, test : %d" % (n_train_females, n_total_females - n_train_females))


    for gender, n_train in zip([males, females], [n_train_males, n_train_females]):
        for i, speaker in enumerate(gender):
            print("%dth speaker processing..." % i)
            speaker_path = os.path.join(hp.unprocessed_data,
                                        speaker, hp.data.contents, '*.wav')
            audio_path = glob.glob(speaker_path)
            mcep = wav_to_mcep(audio_path)

            if i < n_train:  # save mcep as numpy file
                np.save(os.path.join(hp.data.train_path, f"{speaker}.npy"), mcep)
            else:
                np.save(os.path.join(hp.data.test_path, f"{speaker}.npy"), mcep)

def wav_to_mcep(audio_path):
    utterances_spec = []

    for i, utter_path in enumerate(audio_path):
        wav, source_sr = librosa.load(utter_path, sr=None)
        wav = librosa.resample(wav, source_sr, hp.data.sr).astype(np.float) # Resample the wav

        fo, mcep = get_para(wav, fs=hp.data.sr)
        mask = fo.astype(np.bool)
        mcep = mcep[mask, 1:]

        utterances_spec.append(mcep)

    utterances_spec = np.concatenate(utterances_spec, axis=0)
    print(utterances_spec.shape)
    return utterances_spec



if __name__ == "__main__":
    main()