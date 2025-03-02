import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from hydra import compose, initialize
import numpy as np
from omegaconf import OmegaConf
import os, sys
import math
import time
import selfies as sf

from BO.initializer import Initializer
from BO.model.train_bo import update_surr_model, update_models_end_to_end
from BO.model.utils.turbo import generate_batch, update_state
from tqdm import tqdm

def pad(lst, val=1):
    maxlen = max([x.shape[-1] for x in lst])
    return [torch.nn.functional.pad(x, (0, maxlen-x.shape[-1]), value=val) for x in lst]
def cat(lst, val=1):
    return torch.cat(pad(lst, val), 0)
def check(x, y, val=1):
    x, y = pad([x, y], val)
    return (x[:,None,...]==y[None,:,...]).all(-1)

    

class Optimizer(Initializer):
    def __init__(self, **cfg):
        self.__dict__.update(cfg)
        cfg = OmegaConf.create(cfg)
        self.cfg = cfg
        
        print(OmegaConf.to_yaml(cfg))
        
        super(Optimizer, self).__init__(**cfg)

        self.best_score = torch.max(self.init_y)
        self.best_x = self.init_x[torch.argmax(self.init_y)]
        
        self.progress_fails_since_last_e2e = 0
        self.tot_num_e2e_updates = 0
        self.initial_model_training_complete = False
        self.new_best_found = False
        self.num_calls = 0
        
    def selfies_to_txt(self, x):
        import selfies as sf
        if x.dim() == 2:
            return [sf.decoder(self.objective.dataobj.decode(xx)) for xx in x]
        else:
            return sf.decoder(self.objective.dataobj.decode(x))
        
    def txt_to_selfies(self, x):
        return self.objective.dataobj.encode(self.objective.dataobj.tokenize_selfies([sf.encoder(x)])[0])
    
    def run_NFBO(self): 
        self.vae_optim = torch.optim.Adam([{'params': self.vae_model.parameters(), 'lr': self.vae_lr}])
        self.surr_optim = torch.optim.Adam([{'params': self.surrogate_model.parameters(), 'lr': self.surr_lr}])
        
        self.now_train_oracles = 0
        if (self.cfg.objective.name)=='guacamol' and (self.cfg.generative_model.name)=='seqflow':
            self.vae_model.set_dataobj(self.objective.dataobj)
        self.update_surrogate_model() 

        while self.num_calls < self.cfg.max_n_oracle_calls:
            self.log_data_to_wandb_on_each_loop()
                
            if self.now_train_oracles >= 0:
                self.update_models_e2e(self.cfg.wandb, self.tracker)
                self.now_train_oracles -= self.train_oracle
            
            
            self.acquisition()

            if self.tr_state.restart_triggered:
                self.initialize_tr_state()
            if self.new_best_found:
                if self.cfg.verbose:
                    print("\nNew best found:")
                    self.print_progress_update()
                self.new_best_found = False
                
        if self.cfg.verbose:
            print("\nOptimization Run Finished, Final Results:")
            self.print_progress_update()

        self.log_topk_table_wandb()

        return self 

    def check_oracle(self, input_x):
        str_x = str(input_x.tolist())
        if str_x in self.xs_to_scores_dict:
            y = self.xs_to_scores_dict[str_x]
        else:
            y = self.objective.run_oracle(input_x)
            self.xs_to_scores_dict[str_x] = y
        return y
    
    def update_next(self, z_next_, y_next_, x_next_, acquisition=False, acq_len=None):
        z_next_ = z_next_.detach().cpu() 
        y_next_ = y_next_.detach().cpu()
        x_next_ = x_next_.detach().cpu()
        if len(y_next_.shape) > 1:
            y_next_ = y_next_.squeeze() 
        if len(z_next_.shape) == 1:
            z_next_ = z_next_.unsqueeze(0)
        if x_next_.ndim == 1:
            x_next_ = x_next_[None, ...]
        progress = False

        isin = check(x_next_, self.init_x).any(-1)
        x_next_ = x_next_[~isin]
        y_next_ = y_next_[~isin]
        z_next_ = z_next_[~isin]
        
        self.num_calls += len(y_next_)
        print(f"new call {len(y_next_)}")
        
        self.init_x = cat((self.init_x, x_next_))
        self.init_y = torch.cat((self.init_y, y_next_))
        self.init_z = torch.cat((self.init_z, z_next_))
        
        
        self.top_k_ys = torch.cat((self.top_k_ys, y_next_))
        topidx = self.top_k_ys.argsort(descending=True)[:self.k]
        self.top_k_ys = self.top_k_ys[topidx]
        
        self.top_k_xs = cat((self.top_k_xs, x_next_))[topidx] 
        self.top_k_zs = torch.cat((self.top_k_zs, z_next_.cuda()))[topidx] 

        if self.top_k_ys.max() > self.best_score:
            self.progress_fails_since_last_e2e = 0
            progress = True
            self.best_score = self.top_k_ys.max()     
            self.best_x = torch.tensor(self.top_k_xs[self.top_k_ys.argmax()])
            self.new_best_found = True
            
        
        if (not progress) and acquisition:
            self.progress_fails_since_last_e2e += 1

        y_next_ = y_next_.unsqueeze(-1)
        
        isin_ratio = 1-(len(x_next_)/acq_len)
        if isin_ratio>0.95:
            self.tr_state.length = self.tr_state.length * 1.2
        elif isin_ratio<0.8:
            self.tr_state.length = self.tr_state.length / 1.2
        
        raw_max_y = self.raw_init_y.numpy().max(0)
        raw_argmax_y = self.raw_init_y.numpy().argmax(0)
        max_y = self.init_y.numpy().max(0)
        argmax_y = self.init_y.numpy().argmax(0)   
        print(f"len_x:{len(self.init_x):5d}, raw: {raw_max_y}, raw_idx:{raw_argmax_y}, y: {max_y:.3f} [{argmax_y}],")
        if len(y_next_) > 0:
            print(f"# find - min:{y_next_.min().item():.3f}, max:{y_next_.max().item():.3f} find:{(1-isin_ratio)*100:.1f}%")

        self.now_train_oracles += len(y_next_)
        return self


    def update_models_e2e(self, wandb, tracker):
        self.progress_fails_since_last_e2e = 0
        new_xs = self.init_x[-self.newd:] if self.newd>0 else self.init_x[:0]
        new_ys = self.init_y[-self.newd:] if self.newd>0 else self.init_y[:0]
        train_x = cat((new_xs, self.top_k_xs))
        train_y = torch.cat([new_ys, self.top_k_ys], 0)
        print(f"update_e2e, train_x: {train_x.shape}, train_y: {train_y.shape}")
        
        print(f'{self.num_calls} oracles has been called.')
        train_vae_epochs = self.vae_epochs if self.initial_model_training_complete else self.init_vae_epochs
        present_vae_epochs = int(self.tot_num_e2e_updates+train_vae_epochs) - int(self.tot_num_e2e_updates)
        self.tot_num_e2e_updates += train_vae_epochs
        if train_vae_epochs > 0 and present_vae_epochs > 0:
            self.vae_model, self.surrogate_model = update_models_end_to_end(
                self.cfg, 
                self.init_x,
                self.init_y, 
                train_x,
                train_y,
                self.vae_model,
                self.surrogate_model,
                self.vae_optim,
                self.surr_optim,
                present_vae_epochs,
                wandb,
                tracker,
                self.train_bsz,
            )
            
            with torch.no_grad():
                topkz, logp, _ = self.vae_model.encode_z(self.top_k_xs.cuda())
                self.top_k_zs = topkz.reshape(len(topkz), -1)
                recentz, logp, _ = self.vae_model.encode_z(self.init_x[-self.bsz:].cuda())
                self.init_z[-self.bsz:] = recentz.reshape(len(recentz), -1)
                
        self.initial_model_training_complete = True

        return self

    def update_surrogate_model(self): 
        
        
        if not self.initial_model_training_complete:
            n_epochs = self.init_surr_epochs
            train_x = self.init_x.cuda()
            train_y = self.init_y.cuda()
            train_z = self.init_z.cuda()
        else:
            n_epochs = self.surr_epochs
            train_x = self.init_x[-self.newd:].cuda() if self.newd>0 else self.init_x[:0].cuda()
            train_y = self.init_y[-self.newd:].cuda() if self.newd>0 else self.init_y[:0].cuda()
            train_z = self.init_z[-self.newd:].cuda() if self.newd>0 else self.init_z[:0].cuda()
            
            train_x = cat([train_x, self.top_k_xs.cuda()])
            train_y = torch.cat([train_y, self.top_k_ys.cuda()], 0)
            train_z = torch.cat([train_z, self.top_k_zs.cuda()], 0)
            
        self.vae_model.eval()
        self.surrogate_model = update_surr_model(
            self.cfg, 
            self.surrogate_model,
            self.surr_optim,
            train_z,
            train_y,
            n_epochs,
            self.wandb,
            self.tracker,
        )

        return self



    def acquisition(self):
        '''Generate new candidate points, 
        evaluate them, and update data
        '''
        print("Acquisition")
        self.vae_model.eval()
        
        with torch.no_grad():
            if self.temperature > 0:
                mask = torch.ones_like(self.init_y).bool()
                invidx = torch.arange(len(self.init_y))
                topidx = []
                def minmax_norm(tsr):
                    if tsr.max()==tsr.min():
                        return tsr
                    return (tsr - tsr.min())/(tsr.max()-tsr.min())
                for i in range(self.acq_topk):
                    p = torch.nn.functional.softmax(minmax_norm(self.init_y[mask]) / self.temperature, dim=-1)
                    nowidx = invidx[mask][torch.multinomial(p, 1, replacement=False)]
                    mask[nowidx]=False
                    topidx.append(nowidx)
                topidx = torch.cat(topidx)
                
            else:
                topidx = self.init_y.sort(descending=True)[1][0:self.acq_topk]
            maxx = self.init_x[topidx]
            maxy = self.init_y[topidx][None,...]
            if self.inverse:
                maxz, logp, _ = self.vae_model.encode_z(maxx.cuda())
            else:
                maxz = self.init_z[topidx]; logp=None
                
            maxz = maxz.reshape(len(maxz), -1)
            pass
            
            center = maxz.clone() 
            print(f"now tr length = {self.tr_state.length}")
            
            
            # TACS
            if self.dacs_temp != 0:
                ref_z = maxz.clone() 
                B = 1
                N,T = maxx.shape
                ref_z = ref_z.reshape(*maxx.shape,-1)

                ref_z = ref_z.repeat(B, T, 1, 1, 1)
                ref_z[:, torch.arange(T), :, torch.arange(T), :]=torch.randn_like(ref_z[:, torch.arange(T), :, torch.arange(T), :])  ## B, T, N, T, F
                ref_z = ref_z.reshape(B, T*N,-1)
                fullx = torch.full((B,T*N,T), torch.nan).long().to(ref_z.device)
                def smoothing(inp):
                    epsilon = 0.1
                    return (inp * (1 - epsilon)) + (epsilon / 2)
                def logmean(inp):
                    return inp.max(0).values + torch.log(torch.exp(inp - inp.max(0).values).sum(0)) - np.log(inp.shape[0])
                for bb in range(B):
                    xhat, valid_mask, logp_z = self.vae_model.decode_z(ref_z[bb])
                    fullx[bb, valid_mask] = xhat
                fullx = fullx.reshape(B,T,N,T)
                same = (fullx == maxx.to(fullx.device)).float()
                smoothed_same = smoothing(same)
                numerator = torch.full_like(same, 1)
                smoothed_numerator = smoothing(numerator)
                log_numerator = logmean(torch.log(smoothed_numerator).sum(-1))
                log_same = logmean(torch.log(smoothing(same)).sum(-1))
                scale = (log_numerator - log_same).T
            else:
                scale = None
                
            
            z_next, z_cand, z_check = generate_batch(
                z_center=center,
                state=self.tr_state,
                model=self.surrogate_model,
                X=self.init_x,
                Y=self.init_y,
                Z=self.init_z,
                batch_size=self.acq_bsz, 
                acqf=self.acq_func,
                vae_model=self.vae_model,
                min_prob=self.min_prob,
                max_prob=self.max_prob,
                logp=logp,
                scale=scale,
                dacs_temp=self.dacs_temp,
            )
            
            print(f"{maxy.max().item():.3f}: {self.selfies_to_txt(maxx[maxy.argmax()])}")
            
            x_next, valid_mask, logp_z = self.vae_model.decode_z(z_next, sampling=False)
            z_next = z_next[valid_mask]
            acq_size = len(x_next)
            print(f"acquisition x_next: {len(x_next)}:")
            x_next, ind = torch.unique(x_next, sorted=False, return_inverse=True, dim=0)
            print(f"unique x_next: {len(x_next)}:")
            rev_ind = torch.zeros(len(x_next), device=x_next.device).long().scatter_(0, ind, torch.arange(len(ind), device=x_next.device))
            z_next = z_next[rev_ind]
            y_next = self.check_oracle(x_next.cpu())
            if len(y_next)>0:
                print(f"acquisition x_next: {len(x_next)}: {y_next.min():.3f}~{y_next.max():.3f}")
            
        if y_next != None:
            self.update_next(
                z_next,
                y_next,
                x_next,
                acquisition=True,
                acq_len=self.acq_bsz*self.acq_topk,
            )
            if self.wandb:
                dict_log = {
                    "n_oracle_calls": self.num_calls,
                }
                if len(y_next)>0:
                    dict_log.update({'find' : y_next.max().item(),})
                self.tracker.log(dict_log)
            return x_next, y_next
        else:
            self.progress_fails_since_last_e2e += 1
            if self.cfg.verbose:
                print("GOT NO VALID Y_NEXT TO UPDATE DATA, RERUNNING ACQUISITOIN...")
            return None
                
                
def main(**argv):
    initialize(version_base=None, config_path="../config")
    cfg = compose(config_name="optimizer", overrides=sys.argv[1:])
    optimizer = hydra.utils.instantiate(cfg, _recursive_=False)
    return optimizer.run_NFBO()
    
if __name__=="__main__":
    main()