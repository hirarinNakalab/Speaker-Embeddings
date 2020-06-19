import os
import torch
import warnings
warnings.simplefilter('ignore')
from torch.utils.data import DataLoader

from hparam import hparam as hp
from dataloader import JVSNonpara
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

    train_dataset = JVSNonpara(spekers_dict, device, net)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False,
                              num_workers=hp.train.num_workers, drop_last=True)

    simmat_loss = SimMatrixLoss(device, sim_csv_path=sim_csv_path)
    optimizer = torch.optim.Adagrad(net.parameters(), lr=hp.train.lr)
    
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
    
    net.train()
    iteration = 0
    print('=' * 30)
    for e in range(hp.train.iteration):
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
                mesg = f"Iteration:{iteration}\t" \
                       f"Loss:{loss:.4f}\tTotal Loss:{total_loss / (batch_id + 1):.4f}"
                print(mesg)

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


            

if __name__=="__main__":
    train()