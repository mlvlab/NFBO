import torch
import math
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import functools
from scipy import stats
import argparse
import pdb
from tqdm import tqdm


def update_models_end_to_end(
    cfg,       
    all_x,
    all_y, 
    train_x,
    train_y,
    vae_model,
    surrogate_model,
    vae_optim,
    surr_optim,
    num_update_epochs,
    wandb,
    tracker,
    train_bsz,
):
    vae_model.train()
    surrogate_model.train()
    bsz = train_bsz

    tot_vae_loss = []
    tot_diff = []
    out_dict = {}
    
    
    print("Train VAE model")
    train_dataset = TensorDataset(torch.tensor(train_x).cuda(), torch.tensor(train_y).float().cuda())
    train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
    for _ in tqdm(range(num_update_epochs)):
        for it, (batch_list, batch_y) in enumerate(train_loader):
            
            _, vae_loss, out_dict = vae_model(batch_list)
            z, _, _ = vae_model.encode_z(batch_list, sampling=False)
            z = z.reshape(z.shape[0], -1)
            diff = torch.tensor(-1.)

            loss = vae_loss
            
            tot_diff.append(diff.mean().item())
            tot_vae_loss.append(vae_loss.item())
            
            vae_optim.zero_grad()
            surr_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_model.parameters(), max_norm=1.0)
            vae_optim.step()
            surr_optim.step()
            
    
    tot_diff = sum(tot_diff) / len(tot_diff) if len(tot_diff)!=0 else 0            
    tot_vae_loss = sum(tot_vae_loss) / len(tot_vae_loss) if len(tot_vae_loss)!=0 else 0
    print(f"e2e, vae_loss:{tot_vae_loss:.3f}")
    if wandb:
        dict_log = {
            'e2e_diff' : tot_diff,
            'e2e_vae_loss' : tot_vae_loss,
        }
        dict_log.update(out_dict)
        tracker.log(dict_log)
    vae_model.eval()
    surrogate_model.eval()

    return vae_model, surrogate_model


def update_surr_model(
    cfg, 
    model,
    surr_optim,
    train_z,
    train_y,
    n_epochs,
    wandb,
    tracker,
):
    print("update surr model")
    model = model.train()
    train_bsz = min(len(train_y),128)
    train_dataset = TensorDataset(train_z.cuda(), train_y.cuda())
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for _ in tqdm(range(n_epochs)):
        tot_loss = []
        tot_meandiff = []
        tot_std = []
        for (inputs, scores) in train_loader:
            output = model(inputs.cuda())
            loss = model.loss(output, scores.cuda())        
            surr_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            surr_optim.step()
            tot_loss.append(loss.item())
            tot_meandiff.append((scores - output.mean).abs().mean().item())
            tot_std.append(output.variance.pow(1/2).mean().item())
        tot_loss = sum(tot_loss) / len(tot_loss)
        tot_meandiff = sum(tot_meandiff) / len(tot_meandiff)
        tot_std = sum(tot_std) / len(tot_std)
        print(f"surr_loss {tot_loss:.3f}, meandiff {tot_meandiff:.3f}, datadiff {(train_y - train_y.mean()).abs().mean().item():.3f}, std {tot_std:.3f}")
    model = model.eval()
    
    if wandb and n_epochs>0:
        dict_log = {
            'surr_loss' : tot_loss,
        }
        tracker.log(dict_log)

    return model     
    print("update surr model end") 
      

