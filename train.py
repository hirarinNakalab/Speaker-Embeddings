import os
import random
import torch
import warnings
warnings.simplefilter('ignore')
from torch.utils.data import DataLoader

from hparam import hparam as hp
from dataloader import JVSDataset
from model import FFNet, SimMatrixLoss
from preprocess import get_speakers_dict



def train():
    device = torch.device(hp.device)

    net = FFNet().to(device)
    if hp.train.restore:
        net.load_state_dict(torch.load(model_path=None))

    gender = "female"
    sim_csv_path = hp.data.sim_csv_path.format(gender)
    spekers_dict = get_speakers_dict()[gender]

    train_dataset = JVSDataset(spekers_dict, device, net)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False,
                              num_workers=hp.train.num_workers, drop_last=True)

    simmat_loss = SimMatrixLoss(device, sim_csv_path=sim_csv_path)
    optimizer = torch.optim.Adagrad(net.parameters(), lr=hp.train.lr)
    
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
    
    net.train()
    iteration = 0
    print('=' * 30)
    for e in range(hp.train.epochs):
        total_loss = 0
        for batch_id, d_vectors in enumerate(train_loader):
            #gradient accumulates
            optimizer.zero_grad()

            #get loss, call backward, step optimizer
            loss = simmat_loss(d_vectors)

            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss

            iteration += 1
            if (batch_id + 1) % hp.train.log_interval == 0:
                mesg = f"Epoch:{e + 1}[{batch_id + 1}/{len(train_dataset) // hp.train.N}]," \
                       f"Iteration:{iteration}\t" \
                       f"Loss:{loss:.4f}\tTotal Loss:{total_loss / (batch_id + 1):.4f}"
                print(mesg)

        # print('='*30)

        if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
            net.eval().cpu()
            ckpt_model_filename = f"ckpt_epoch_{e+1}.pth"
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            torch.save(net.state_dict(), ckpt_model_path)
            net.to(device).train()

    #save model
    net.eval().cpu()
    save_model_filename = f"final_epoch_{e + 1}.model"
    save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
    torch.save(net.state_dict(), save_model_path)
    
    print("\nDone, trained model saved at", save_model_path)

def main(model_path):
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
            

if __name__=="__main__":
    train()