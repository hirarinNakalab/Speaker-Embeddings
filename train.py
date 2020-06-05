import os
import random
import time
import torch
from torch.utils.data import DataLoader

from hparam import hparam as hp
from dataloader import JVSDataset, JVSDatasetPreprocessed
from model import FFNet, SimMatrixLoss
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
from preprocess import get_speakers_dict



def train():
    device = torch.device(hp.device)

    net = FFNet().to(device)
    if hp.train.restore:
        net.load_state_dict(torch.load(model_path=None))

    gender = "female"
    spekers_dict = get_speakers_dict()[gender]

    if hp.data.data_preprocessed:
        train_dataset = JVSDatasetPreprocessed(
            spekers_dict=spekers_dict, device=device, model=net)
    else:
        train_dataset = JVSDataset()

    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True,
                              num_workers=hp.train.num_workers, drop_last=True)

    # simmat_loss = SimMatrixLoss(device)

    optimizer = torch.optim.Adagrad(net.parameters(), lr=hp.train.lr)
    
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
    
    net.train()
    iteration = 0
    for e in range(hp.train.epochs):
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            utters_list, selected_speaker = batch
            mel_db_batch = mel_db_batch.to(device)
            
            mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
            perm = random.sample(range(0, hp.train.N*hp.train.M), hp.train.N*hp.train.M)
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]
            #gradient accumulates
            optimizer.zero_grad()
            
            embeddings = net(mel_db_batch)
            embeddings = embeddings[unperm]
            embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))
            
            #get loss, call backward, step optimizer
            loss = ge2e_loss(embeddings) #wants (Speaker, Utterances, embedding)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            optimizer.step()
            
            total_loss = total_loss + loss
            iteration += 1
            if (batch_id + 1) % hp.train.log_interval == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(time.ctime(), e+1,
                        batch_id+1, len(train_dataset)//hp.train.N, iteration,loss, total_loss / (batch_id + 1))
                print(mesg)
                if hp.train.log_file is not None:
                    with open(hp.train.log_file,'a') as f:
                        f.write(mesg)
                    
        if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
            net.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(e+1) + "_batch_id_" + str(batch_id+1) + ".pth"
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            torch.save(net.state_dict(), ckpt_model_path)
            net.to(device).train()

    #save model
    net.eval().cpu()
    save_model_filename = "final_epoch_" + str(e + 1) + "_batch_id_" + str(batch_id + 1) + ".model"
    save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
    torch.save(net.state_dict(), save_model_path)
    
    print("\nDone, trained model saved at", save_model_path)

def main(model_path):
    
    if hp.data.data_preprocessed:
        test_dataset = JVSDatasetPreprocessed()
    else:
        test_dataset = JVSDataset()

    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True,
                             num_workers=hp.test.num_workers, drop_last=True)
    
    net = FFNet()
    net.load_state_dict(torch.load(model_path))
    net.eval()
    
    avg_EER = 0
    for e in range(hp.test.epochs):
        batch_avg_EER = 0
        for batch_id, mel_db_batch in enumerate(test_loader):
            assert hp.test.M % 2 == 0
            enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1)/2), dim=1)
            
            enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*hp.test.M//2, enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (hp.test.N*hp.test.M//2, verification_batch.size(2), verification_batch.size(3)))
            
            perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
                
            verification_batch = verification_batch[perm]
            enrollment_embeddings = net(enrollment_batch)
            verification_embeddings = net(verification_batch)
            verification_embeddings = verification_embeddings[unperm]
            
            enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, hp.test.M//2, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, hp.test.M//2, verification_embeddings.size(1)))
            
            enrollment_centroids = get_centroids(enrollment_embeddings)
            
            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)
            
    print("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))
        
if __name__=="__main__":
    train()