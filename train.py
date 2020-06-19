import os
import sys

import torch
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
from torch.utils.data import DataLoader

from hparam import hparam as hp
from dataloader import JVSNonparaTrain, JVSNonparaVal
from model import FFNet, SimMatrixLoss
from preprocess import get_speakers_dict



def train(gender="female"):
    device = torch.device(hp.device)

    net = FFNet().to(device)
    if hp.train.restore:
        net.load_state_dict(torch.load(model_path=None))

    sim_csv_path = hp.data.sim_csv_path.format(gender)
    spekers_dict = get_speakers_dict()[gender]

    train_dataset = JVSNonparaTrain(spekers_dict, device, net)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False,
                              num_workers=hp.train.num_workers, drop_last=True)
    val_dataset = JVSNonparaVal(spekers_dict, device, net)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False,
                              num_workers=hp.test.num_workers, drop_last=True)

    simmat_loss = SimMatrixLoss(device, sim_csv_path=sim_csv_path)
    optimizer = torch.optim.Adagrad(net.parameters(), lr=hp.train.lr)
    
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
    
    net.train()
    iteration = 0
    print('=' * 30)
    losses = []
    for i in range(hp.train.iteration):
        total_loss = 0
        for batch_id, d_vectors in enumerate(train_loader):
            #gradient accumulates
            optimizer.zero_grad()

            #get loss, call backward, step optimizer
            loss = simmat_loss(d_vectors)

            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss
            losses.append(loss.item())
            iteration += 1
            if (batch_id + 1) % hp.train.log_interval == 0:
                mesg = f"Iteration:{iteration}\t" \
                       f"Loss:{loss:.4f}\tTotal Loss:{total_loss / (batch_id + 1):.4f}"
                print(mesg, end="\t")

        for batch_id, d_vectors in enumerate(val_loader):
            with torch.no_grad():
                loss = simmat_loss(d_vectors)
                if (batch_id + 1) % hp.train.log_interval == 0:
                    print(f"Val Loss:{loss:.4f}\t")

        if hp.train.checkpoint_dir is not None and (i + 1) % hp.train.checkpoint_interval == 0:
            net.eval().cpu()
            ckpt_model_filename = f"ckpt_epoch_{i+1}.pth"
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            torch.save(net.state_dict(), ckpt_model_path)
            net.to(device).train()

    #save model
    net.eval().cpu()
    save_model_filename = f"final_epoch_{i + 1}.model"
    save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
    torch.save(net.state_dict(), save_model_path)
    
    print("\nDone, trained model saved at", save_model_path)
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

            

if __name__=="__main__":
    if len(sys.argv) > 1:
        gender = sys.argv[1]
    else:
        sys.exit()
    train(gender=gender)