import os
import sys
import math
import random

import numpy as np
from scipy import stats
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import hydra
import wandb

from hydra import compose, initialize
from omegaconf import OmegaConf

from data.data_load import DataStorage
from BO.model.utils.turbo import TurboState

def pad(lst, val=1):
    maxlen = max([x.shape[-1] for x in lst])
    return [torch.nn.functional.pad(x, (0, maxlen-x.shape[-1]), value=val) for x in lst]
def cat(lst, val=1):
    return torch.cat(pad(lst, val), 0)
def check(x, y, val=1):
    x, y = pad([x, y], val)
    return (x[:,None,...]==y[None,:,...]).all(-1)

class Initializer(object):
    def __init__(self, **cfg):
        self.__dict__.update(cfg)
        cfg = OmegaConf.create(cfg)
        self.cfg = cfg
        
        if cfg.gpu >= 0:
            torch.cuda.set_device(f"cuda:{cfg.gpu}")
        
        self.create_wandb_tracker()
        
        if cfg.wandb:
            os.makedirs(cfg.log_dir, exist_ok = True)
            
        self.initialize_objective_func(cfg.objective)
        
        self.partial_vae_model = hydra.utils.instantiate(cfg.generative_model, _recursive_=False)
        self.partial_surrogate_model = hydra.utils.instantiate(cfg.surrogate_model, _recursive_=False)
        
        self.initialize_vae_model(cfg.generative_model)
        
        self.vae_model.set_dataobj(self.objective.dataobj)
        
        self.initialize_data(cfg.data)
        self.train_vae_model()
        self.initialize_z()
        
        self.initialize_surrogate_model(cfg.surrogate_model)
        
        self.initialize_top_k()
        self.initialize_tr_state()
        self.initialize_xs_to_scores_dict()
        
    def initialize_objective_func(self, cfg):
        self.objective = hydra.utils.instantiate(cfg, _recursive_=False)
        
    def initialize_data(self, cfg):
        self.data = hydra.utils.instantiate(cfg, _recursive_=False)
        self.graph, self.init_x, self.raw_init_y, self.all_loader, self.train_loader, self.val_loader, self.test_loader, self.scaler, self.pretrain_x = self.data.make_dataloader(objective = self.objective)
        self.init_y = self.raw_init_y[:,0]
        if self.init_x.dtype in [np.float64, np.float32, np.float16, np.int64, np.int32, np.int16, np.int8, np.uint, np.uint64, np.uint32, np.uint16, np.uint8]:
            self.init_x = torch.tensor(self.init_x)
        
        if self.verbose:
            print("Loaded initial oracle data")
            print("init oracle y shape:", self.init_y.shape)
            print(f"init oracle x list length: {len(self.init_x)}\n")
            print(f"init pretrain x list length: {len(self.pretrain_x)}\n")
        return self

    def initialize_vae_model(self, cfg):
        
        if cfg.name == 'seqflow':
            self.inverse = True
            self.vae_model = self.partial_vae_model(vocab_size=self.objective.dataobj.vocab_size)
        elif cfg.name == 'textflow':
            self.inverse = False
            self.vae_model = self.partial_vae_model(vocab_size=self.objective.dataobj.vocab_size)
            
        self.vae_model.cuda()
        
    def train_vae_model(self):
        train_dataset = TensorDataset(torch.tensor(self.pretrain_x).cuda())
        self.train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            
        path_model = f"{self.cfg.root_dir}/model_weight/{self.cfg.generative_model.name}_{self.cfg.pretrain_name}.pt"
        if self.cfg.use_pretrain:
            if os.path.isfile(path_model):
                state_dict = torch.load(path_model, map_location=torch.device('cuda')) 
                self.vae_model.load_state_dict(state_dict)
                print(f"Load pretrained generative model={self.cfg.pretrain_name}, on {path_model}")
            else:
                print(f"Error not found pretrained generative model={self.cfg.pretrain_name}, on {path_model}, exit!")
                exit()
        else:
            n_epochs = self.cfg.init_vae_epochs
            optimizer = torch.optim.Adam(self.vae_model.parameters(), lr=0.001)
            
            for epoch in tqdm(range(n_epochs)):
                tot_vae_loss = []
                for i, (inputs, ) in enumerate(self.train_loader):
                    z, vae_loss, out_dict = self.vae_model(inputs.cuda())
                    z = z.reshape(z.shape[0], -1)
                    dim = z.shape[-1]
                    
                    optimizer.zero_grad()
                    vae_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.vae_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    tot_vae_loss.append(vae_loss.item())
                    
                tot_vae_loss = sum(tot_vae_loss) / len(tot_vae_loss) if len(tot_vae_loss)!=0 else 0
                
                print(f"epoch={epoch:4d}, loss={tot_vae_loss:.3f}, ", *[f"{k}: {v.item():.3f}, " for k, v in out_dict.items()])
                
                if epoch % 1 == 0:
                    with torch.no_grad():
                        import selfies as sf
                        rx, val, _ = self.vae_model.decode_z(torch.randn(100, dim).cuda())
                        ry = self.objective.run_oracle(rx)
                        print(f"##############################")
                        print(str([sf.decoder(self.objective.dataobj.decode(xx)) for xx in rx[0:5]]))
                        print(f"{self.objective.task_id} data mean: {self.init_y.mean():.4f}, std:{self.init_y.std():.4f}")
                        print(f"epoch:{epoch:3d} mean: {ry.mean().item():.4f}, std: {ry.std().item():.4f}")
                        print(f"##############################")
                        if self.wandb:
                            self.tracker.log({
                                "task_mean": self.init_y.mean().item(),
                                "task_std": self.init_y.std().item(),
                                "gen_rand_task_mean": ry.mean().item(),
                                "gen_rand_task_std": ry.std().item(),
                            })
                        
                if self.cfg.save_pretrain:
                    save_name = path_model[:-3]+f"_new_{epoch}.pt"
                    torch.save(self.vae_model.state_dict(), save_name)
                    print(f"Save pretrained generative model={self.cfg.pretrain_name}, on {save_name}")
        
    def initialize_z(self):
        init_x = []
        init_z = []
        init_y = []
        self.vae_model.eval() 
        
        bsz = 1024
        train_dataset = TensorDataset(torch.tensor(self.init_x).cuda(), torch.tensor(self.init_y).float().cuda())
        train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=False)
        
        print("initialize_z")
        with torch.no_grad():
            for xs_batch, ys_batch in tqdm(train_loader):
                xs_batch = xs_batch.cuda()
                zs, logp, _ = self.vae_model.encode_z(xs_batch, sampling=True)
                xs = xs_batch
                ys = ys_batch
                
                init_z.append(zs.detach().cpu())
                init_x.append(xs.detach().cpu())
                init_y.append(ys.detach().cpu())
            init_z = torch.cat(init_z, dim=0)
            init_z = init_z.reshape(init_z.shape[0], -1)
            init_y = torch.cat(init_y, dim=0)
            init_x = cat(init_x)
        
        self.vae_model.train()
        
        self.init_z = init_z
        self.init_x = init_x
        self.init_y = init_y
        return self.init_z

        
    def initialize_top_k(self):
        self.top_k_ys, top_k_idxs = torch.topk(self.init_y.squeeze(), min(self.k, len(self.init_y)))
        top_k_idxs = top_k_idxs.tolist()
        self.top_k_xs = self.init_x[top_k_idxs]
        self.top_k_ys = self.top_k_ys
        self.top_k_zs = self.init_z[top_k_idxs].cuda()

    def initialize_xs_to_scores_dict(self):
        self.xs_to_scores_dict = dict([(str(x.tolist()), y) for x, y in zip(self.init_x, zip(self.init_y, self.raw_init_y))])

    def initialize_tr_state(self):
        self.tr_state = TurboState(
            prob_perturb=self.prob_perturb,
            dim=self.init_z.shape[-1],
            batch_size=self.bsz, 
            best_value=torch.max(self.init_y).item(),
            length=self.cfg.tr_length,
            )
        return self

    def initialize_surrogate_model(self, cfg):
        n_pts = min(self.init_z.shape[0], 1024)
        self.surrogate_model = self.partial_surrogate_model(self.init_z[:n_pts, :].cuda()).cuda()

        return self


    def create_wandb_tracker(self):                                         
        if self.cfg.wandb:
            self.wandb_entity=self.cfg.wandb_entity
            
            self.tracker = wandb.init(
                project=f"{self.cfg.wandb_project_name}",
                name=f"{self.objective.task_id}_{random.randint(0,10000):04d}" if self.cfg.wandb_name is None else self.cfg.wandb_name,
                config=OmegaConf.to_container(self.cfg),
            ) 
            self.wandb_run_name = wandb.run.name
            self.wandb_run_id = wandb.run.id       
        else:
            os.environ["WANDB_MODE"]="disabled"
            self.tracker = None 
            self.wandb_run_name = 'no-wandb-tracking'
        
        return self

    def log_data_to_wandb_on_each_loop(self):
        if self.cfg.wandb:
            dict_log = {
                "best_found": self.best_score,
                "n_oracle_calls": self.num_calls,
                "total_number_of_e2e_updates": self.tot_num_e2e_updates,
                "best_input_seen": self.best_x,
            }
            dict_log[f"TR_length"] = self.tr_state.length
            self.tracker.log(dict_log)
        return self
    
    def print_progress_update(self):
        if self.cfg.wandb:
            print(f"Optimization Run: {self.cfg.wandb_project_name}, {wandb.run.name}")
        print(f"Best X Found: {self.selfies_to_txt(self.best_x)}")
        print(f"Best {self.cfg.objective.task_id} Score: {self.best_score}")
        print(f"Total Number of Oracle Calls (Function Evaluations): {self.num_calls}")
        return self

    def log_topk_table_wandb(self):
        if self.cfg.wandb:
            cols = ["Top K Scores", "Top K Strings"]
            data_list = []
            for ix, score in enumerate(self.top_k_ys):
                data_list.append([ score, str(self.top_k_xs[ix]) ])
            top_k_table = wandb.Table(columns=cols, data=data_list)
            self.tracker.log({f"top_k_table": top_k_table})
            self.tracker.finish()

        return self