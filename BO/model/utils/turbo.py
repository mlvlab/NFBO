import math
import torch
from dataclasses import dataclass
from torch.quasirandom import SobolEngine
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.generation import MaxPosteriorSampling 


@dataclass
class TurboState:
    prob_perturb: float
    dim: int
    batch_size: int
    length: float = 0.8 
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 32 
    success_counter: int = 0
    success_tolerance: int = 10 
    best_value: float = -float("inf")
    restart_triggered: bool = False

    # def __post_init__(self):
    #     self.failure_tolerance = math.ceil(
    #         max([4.0 / self.batch_size, float(self.dim ) / self.batch_size])
    #     )

def update_state(state, Y_next):
    if state.length_min == None:
        state.length_min = 0.4
    if max(Y_next) > state.best_value + 1e-4 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += len(Y_next)

    if state.success_counter > state.success_tolerance:  # Expand trust region
        state.length *= 2.0
        state.failure_counter = 0
    elif state.failure_counter > state.failure_tolerance and state.failure_tolerance > 0:  # Shrink trust region
        state.length = max(state.length / 2, state.length_min)
        state.success_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length <= state.length_min:
        state.restart_triggered = True
    return state

def generate_batch(
    z_center,
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    Z,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling 
    num_restarts=10,
    raw_samples=256,
    acqf="ts",  # "ei" or "ts"
    dtype=torch.float32,
    device=torch.device('cuda'),
    cand_max = 50000,
    vae_model=None,
    min_prob=0,
    max_prob=1,
    logp=None, 
    scale=None,
    dacs_temp=0,
):
    if logp != None:
        logp=logp[...,0].T
    
    N, F = z_center.shape
    assert acqf in ("ts", "ei", "ucb", "rand")
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None: n_candidates = min(cand_max, max(2000, 200 * Z.shape[-1]))   
    weights = torch.ones_like(z_center)*8 # less than 4 stdevs on either side max 
    
    tr_lb = z_center - weights * state.length / 2.0
    tr_ub = z_center + weights * state.length / 2.0 

    prob_perturb = state.prob_perturb
    
    if acqf == "ei":
        try:
            ei = qExpectedImprovement(model.cuda(), Y.max().cuda() ) 
            z_next, _ = optimize_acqf(ei,bounds=torch.stack([tr_lb, tr_ub]).cuda(),q=batch_size, num_restarts=num_restarts,raw_samples=raw_samples,)
        except: 
            acqf = 'ts'
        x_next = vae_model.decode_z(z_next[...,None], sampling=True)
        z_cand = None
        x_cand = None
    else:
        K=0
        logp_scaler = 1
        
        if dacs_temp > 0:
            logp_p = (scale/dacs_temp).softmax(-1)*scale.shape[-1]
            logp_scaler = logp_p[:,:,None].repeat(1,1,vae_model.zsize)
            logp_scaler = logp_scaler.reshape(N, 1, -1)
            
        n_candidates = n_candidates//N
        dim = Z.shape[-1]
        tr_lb = tr_lb.cuda()[:, None, :]
        tr_ub = tr_ub.cuda()[:, None, :]
        sobol = SobolEngine(dim, scramble=True) 
        pert = sobol.draw(N * n_candidates).to(dtype=dtype).cuda().reshape(N, n_candidates, dim)
        pert = tr_lb + (tr_ub - tr_lb) * pert
        tr_lb = tr_lb.cuda()
        tr_ub = tr_ub.cuda() 
        mask = (torch.rand(N, n_candidates, dim, dtype=dtype, device=device)/(logp_scaler) <= prob_perturb)
        mask = mask.cuda()
        z_cand = z_center[:,None,:].expand(N, n_candidates, dim).clone()
        z_cand = z_cand.cuda()
        z_cand[mask] = pert[mask]
            
        if acqf == "ts":
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False) 
            z_next = thompson_sampling(z_cand.cuda(), num_samples=batch_size)
            z_next = z_next.reshape(-1,dim)
        elif acqf == "ucb":
            pred = model(z_cand.reshape(N*n_candidates, dim))
            acq = (pred.mean + pred.variance).reshape(N, n_candidates)
            val, ind = acq.sort(dim=-1, descending=True)
            z_next = z_cand.gather(1, ind[:,:batch_size].unsqueeze(-1).repeat(1,1,dim))
            z_next = z_next.reshape(-1,dim)
        z_check = None
    return z_next, z_cand, logp_scaler
