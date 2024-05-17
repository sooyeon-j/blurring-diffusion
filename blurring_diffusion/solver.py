import torch
import numpy as np
import networkx as nx
import abc
import torch_dct as dct
from tqdm import trange

from losses import get_score_fn
from utils.graph_utils import mask_adjs, mask_x, gen_noise
from sde import VPSDE, subVPSDE


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t, flags):
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""
  def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.scale_eps = scale_eps
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t, flags):
    pass


class EulerMaruyamaPredictor(Predictor):
  def __init__(self, obj, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.obj = obj

  def update_fn(self, x, adj, flags, t):
    dt = -1. / self.rsde.N

    if self.obj=='x':
      z = gen_noise(x, flags, sym=False)
      drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=False)
      x_mean = x + drift * dt
      x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
      return x, x_mean

    elif self.obj=='adj':
      z = gen_noise(adj, flags)
      drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=True)
      adj_mean = adj + drift * dt
      adj = adj_mean + diffusion[:, None, None] * np.sqrt(-dt) * z

      return adj, adj_mean

    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class ReverseDiffusionPredictor(Predictor):
  def __init__(self, obj, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.obj = obj

  def update_fn(self, x, adj, flags, t):

    if self.obj == 'x':
      f, G = self.rsde.discretize(x, adj, flags, t, is_adj=False)
      z = gen_noise(x, flags, sym=False)
      x_mean = x - f
      x = x_mean + G[:, None, None] * z
      return x, x_mean

    elif self.obj == 'adj':
      f, G = self.rsde.discretize(x, adj, flags, t, is_adj=True)
      z = gen_noise(adj, flags)
      adj_mean = adj - f
      adj = adj_mean + G[:, None, None] * z
      return adj, adj_mean
    
    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
    self.obj = obj
    pass

  def update_fn(self, x, adj, flags, t):
    if self.obj == 'x':
      return x, x
    elif self.obj == 'adj':
      return adj, adj
    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class LangevinCorrector(Corrector):
  def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
    super().__init__(sde, score_fn, snr, scale_eps, n_steps)
    self.obj = obj

  def update_fn(self, x, adj, flags, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    seps = self.scale_eps

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    if self.obj == 'x':
      for i in range(n_steps):
        grad = score_fn(x, adj, flags, t)
        noise = gen_noise(x, flags, sym=False)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None] * grad
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
      return x, x_mean

    elif self.obj == 'adj':
      for i in range(n_steps):
        grad = score_fn(x, adj, flags, t)
        noise = gen_noise(adj, flags)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        adj_mean = adj + step_size[:, None, None] * grad
        adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
      return adj, adj_mean

    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported")


# -------- PC sampler --------
def get_pc_sampler(sde_x, sde_adj, shape_x, shape_adj, predictor='Reverse', corrector='None', 
                   snr=0.1, scale_eps=1.0, n_steps=1, 
                   probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):

  def pc_sampler(model_x, model_adj, init_flags):

    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor 
    corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector

    predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)

    predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device) 
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device) 
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

      # -------- Reverse diffusion process --------
      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        _x = x
        x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
        adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)

        _x = x
        x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
        adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)
      print(' ')
      return (x_mean if denoise else x), (adj_mean if denoise else adj), diff_steps * (n_steps + 1)
  return pc_sampler


# -------- S4 solver --------
def S4_solver(sde_x, sde_adj, shape_x, shape_adj, predictor='None', corrector='None', 
                        snr=0.1, scale_eps=1.0, n_steps=1, 
                        probability_flow=False, continuous=False,
                        denoise=True, eps=1e-3, device='cuda'):

  def s4_solver(model_x, model_adj, init_flags):

    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device) 
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device) 
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)
      dt = -1. / diff_steps

      # -------- Rverse diffusion process --------
      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t
        vec_dt = torch.ones(shape_adj[0], device=t.device) * (dt/2) 

        # -------- Score computation --------
        score_x = score_fn_x(x, adj, flags, vec_t)
        score_adj = score_fn_adj(x, adj, flags, vec_t)

        Sdrift_x = -sde_x.sde(x, vec_t)[1][:, None, None] ** 2 * score_x
        Sdrift_adj  = -sde_adj.sde(adj, vec_t)[1][:, None, None] ** 2 * score_adj

        # -------- Correction step --------
        timestep = (vec_t * (sde_x.N - 1) / sde_x.T).long()

        noise = gen_noise(x, flags, sym=False)
        grad_norm = torch.norm(score_x.reshape(score_x.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        if isinstance(sde_x, VPSDE):
          alpha = sde_x.alphas.to(vec_t.device)[timestep]
        else:
          alpha = torch.ones_like(vec_t)
      
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None] * score_x
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps

        noise = gen_noise(adj, flags)
        grad_norm = torch.norm(score_adj.reshape(score_adj.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        if isinstance(sde_adj, VPSDE):
          alpha = sde_adj.alphas.to(vec_t.device)[timestep] # VP
        else:
          alpha = torch.ones_like(vec_t) # VE
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        adj_mean = adj + step_size[:, None, None] * score_adj
        adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps

        # -------- Prediction step --------
        x_mean = x
        adj_mean = adj
        mu_x, sigma_x = sde_x.transition(x, vec_t, vec_dt)
        mu_adj, sigma_adj = sde_adj.transition(adj, vec_t, vec_dt) 
        x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
        adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)
        
        x = x + Sdrift_x * dt
        adj = adj + Sdrift_adj * dt

        mu_x, sigma_x = sde_x.transition(x, vec_t + vec_dt, vec_dt) 
        mu_adj, sigma_adj = sde_adj.transition(adj, vec_t + vec_dt, vec_dt) 
        x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
        adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)

        x_mean = mu_x
        adj_mean = mu_adj
      print(' ')
      return (x_mean if denoise else x), (adj_mean if denoise else adj), 0
  return s4_solver
  
def backward(sde_x, sde_adj, shape_x, shape_adj, continuous = True, eps = 1e-3, device = 'cuda'):
  def denoise(model_x, model_adj, init_flags):
    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)
    
    # eigenvalue = eigenvalue[None, :].to(device)
    # eigenvector = eigenvector.expand(shape_adj).to(device)
    # eigenvector = mask_adjs(eigenvector, init_flags)
    
    with torch.no_grad():
      x = sde_x.prior_sampling(shape_x).to(device)
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, 0, diff_steps, device=device)

  #     for i in trange(0, diff_steps - 1, desc = '[Sampling]', position = 1, leave=False):
  #       t = timesteps[i]
  
  #       vec_t = torch.ones(shape_adj[0], device=t.device) * t
  #       vec_t_minus_1 = torch.ones(shape_adj[0], device=t.device) * timesteps[i + 1]
        
  #       _x = x
        
  #       alpha_s, sigma_s = sde_x.alpha_sigma(eigenvalue, vec_t_minus_1)
  #       alpha_t, sigma_t = sde_x.alpha_sigma(eigenvalue, vec_t)

  #       x_eps_hat = score_fn_x(x, adj, flags, vec_t, eigenvalue) * sigma_t[:, None, None]
        
  #       x = compute_reverse(x, alpha_s, sigma_s, alpha_t, sigma_t, x_eps_hat, eigenvector, flags, adj = False)
        
  #       alpha_s, sigma_s = sde_adj.alpha_sigma(eigenvalue, vec_t_minus_1)
  #       alpha_t, sigma_t = sde_adj.alpha_sigma(eigenvalue, vec_t)
        
  #       adj_eps_hat = score_fn_adj(_x, adj, flags, vec_t, eigenvalue) * sigma_t[:, None, None]
        
  #       adj = compute_reverse(adj, alpha_s, sigma_s, alpha_t, sigma_t, adj_eps_hat, eigenvector, flags, adj = True)
          
  #     print(' ')
  #   return x, adj
     
  # def compute_reverse(x, alpha_s, sigma_s, alpha_t, sigma_t, eps_hat, eigenvector, flags, adj = False):
  #   delta = 1e-8

  #   alpha_ts = alpha_t / alpha_s
  #   alpha_st = 1 / alpha_ts
  #   sigma_ts2 = (sigma_t ** 2)[:, None] - ((alpha_ts ** 2) * (sigma_s ** 2)[:, None])

  #   sigma_denoise2 = (sigma_ts2 ** 2) * (sigma_s ** 2)[:, None] / (sigma_t ** 2)[:, None]
  #   coeff1 = alpha_ts * (sigma_s ** 2)[:, None] / (sigma_t ** 2)[:, None]
  #   coeff2 = alpha_st * sigma_ts2 / (sigma_t ** 2)[:, None]
    
  #   coeff1_ = torch.diag_embed(coeff1)
  #   coeff2_ = torch.diag_embed(coeff2)  
    
  #   eigenvector_T = eigenvector.transpose(-1, -2)
  #   u_eps = torch.matmul(eigenvector_T, eps_hat)
  #   u_t = torch.matmul(eigenvector_T, x)
  #   term1 = torch.matmul(eigenvector, torch.matmul(coeff1_, u_t))
  #   term2 = torch.matmul(eigenvector, torch.matmul(coeff2_, (u_t - (sigma_t[:, None, None] * u_eps))))
  #   mu_denoise = term1 + term2
    
  #   if adj:
  #     eps = gen_noise(mu_denoise, flags, sym = True)
  #   else: 
  #     eps = gen_noise(mu_denoise, flags, sym = False)
    
  #   res = mu_denoise + torch.matmul(eigenvector, (torch.sqrt(sigma_denoise2)[:, :, None] * eps))

  #   return res
    
  # return denoise      
    
  #   alpha_ts = alpha_t / alpha_s
  #   alpha_st = 1 / alpha_ts
  #   sigma_ts2 = (sigma_t ** 2)[:, None] - ((alpha_ts ** 2) * (sigma_s ** 2)[:, None]) 
    
  #   tmp1 = sigma_s ** 2
  #   tmp2 = ((sigma_t ** 2)[:, None] / (alpha_ts ** 2)) - (sigma_s ** 2)[:, None]
  #   sigma2_denoise = 1 / ((1 / tmp1[:, None]) + (1 / tmp2))
  #   sigma2_denoise_ = torch.diag_embed(sigma2_denoise)

  #   coeff1 = sigma2_denoise * alpha_ts / sigma_ts2
  #   coeff2 = sigma2_denoise * alpha_st / (sigma_s ** 2)[:, None]

  #   coeff1_ = torch.diag_embed(coeff1)
  #   coeff2_ = torch.diag_embed(coeff2)

  #   eigenvector_T = eigenvector.transpose(-1, -2)
  #   u_eps = torch.matmul(eigenvector_T, eps_hat)
  #   u_t = torch.matmul(eigenvector_T, x)
  #   term1 = torch.matmul(eigenvector, torch.matmul(coeff1_, u_t))
  #   term2 = torch.matmul(eigenvector, torch.matmul(coeff2_, (u_t - (sigma_t[:, None, None] * u_eps))))
  #   mu_denoise = term1 + term2
    
  #   if adj:
  #     eps = gen_noise(mu_denoise, flags, sym = True)
  #   else: 
  #     eps = gen_noise(mu_denoise, flags, sym = False)

  #   res = mu_denoise + torch.matmul(eigenvector, torch.matmul(torch.sqrt(sigma2_denoise_), eps))

  #   return res
    
  # return denoise

  #     eigenvector = mask_adjs(eigenvector, flags)
      
      for i in trange(0, diff_steps - 1, desc = '[Sampling]', position = 1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t
        # vec_t_minus_1 = torch.ones(shape_adj[0], device=t.device) * timesteps[i + 1]
        
        _x = x
        
        noise_x = gen_noise(x, flags, sym = False)
        noise_adj = gen_noise(adj, flags, sym = True)
        
        _, sigma = sde_x.alpha_sigma(torch.zeros(adj.shape[0], adj.shape[1]).to(device), vec_t)
        delta = 1.1 * sigma
        score_x = score_fn_x(x, adj, flags, vec_t)
        x_mean = score_x + x
        x = x_mean + delta[:, None, None] * noise_x
        
        score_adj = score_fn_adj(_x, adj, flags, vec_t)
        adj_mean = score_adj + adj
        adj = adj_mean + delta[:, None, None] * noise_adj
        
  #       # alpha_s, sigma_s = sde_x.alpha_sigma(eigenvalue_x, vec_t_minus_1)
  #       # alpha_t, sigma_t = sde_x.alpha_sigma(eigenvalue_x, vec_t)
        
  #       # x_eps_hat = score_fn_x(x, adj, flags, vec_t)
        
  #       # x = compute_reverse(x, alpha_s, sigma_s, alpha_t, sigma_t, x_eps_hat, eigenvector, flags, adj = False)
        
  #       # alpha_s, sigma_s = sde_adj.alpha_sigma(eigenvalue_adj, vec_t_minus_1)
  #       # alpha_t, sigma_t = sde_adj.alpha_sigma(eigenvalue_adj, vec_t)
        
  #       # adj_eps_hat = score_fn_adj(_x, adj, flags, vec_t)
        
  #       # adj = compute_reverse(adj, alpha_s, sigma_s, alpha_t, sigma_t, adj_eps_hat, eigenvector, flags, adj = True)
        
  #       # import pdb; pdb.set_trace()

  #       ###############################################################################3
  #       # _f, _G = sde_x.discretize(x, vec_t)
        
  #       # score_x = score_fn_x(x, adj, flags)
        
  #       # f = _f - _G[:, None, None] ** 2 * score_x
        
  #       # predict_x_mean = x - f
        
  #       # x = predict_x_mean + _G[:, None, None] * noise_x 
        
  #       # _f, _G = sde_adj.discretize(adj, vec_t)
        
  #       # score_adj = score_fn_adj(_x, adj, flags)
        
  #       # f = _f - _G[:, None, None] ** 2 * score_adj
        
  #       # predict_adj_mean = adj - f
        
  #       # adj = predict_adj_mean + _G[:, None, None] * noise_adj
        
      print(' ')
    return x, adj

  # def compute_reverse(x, alpha_s, sigma_s, alpha_t, sigma_t, eps_hat, eigenvector, flags, adj = False):
  #   alpha_ts = alpha_t / alpha_s # <1
  #   alpha_st = 1 / alpha_ts
  #   sigma_ts2 = (sigma_t ** 2)[:, None, None] - ((alpha_ts ** 2) * (sigma_s ** 2)[:, None, None])
    
  #   # sigma2_denoise = 1 / torch.clip(
  #   #   1 / torch.clip((sigma_s ** 2)[:, None, None], min = delta) +
  #   #   1 / torch.clip((sigma_t ** 2)[:, None, None] / alpha_ts ** 2 - (sigma_s ** 2)[:, None, None], min = delta),
  #   #   min = delta)
    
  #   tmp1 = sigma_s ** 2
  #   tmp2 = ((sigma_t ** 2)[:, None, None] / (alpha_ts ** 2)) - (sigma_s ** 2)[:, None, None]
  #   sigma2_denoise = 1 / ((1 / tmp1[:, None, None]) + (1 / tmp2))
    
  #   coeff1 = sigma2_denoise * alpha_ts / sigma_ts2
  #   coeff2 = sigma2_denoise * alpha_st / (sigma_s ** 2)[:, None, None]

  #   # coeff1 = sigma2_denoise * alpha_ts / sigma_ts2
  #   # coeff2 = sigma2_denoise * alpha_st / torch.clip((sigma_s ** 2)[:, None, None], min = delta)
    
  #   eigenvector_T = eigenvector.transpose(-1, -2)
  #   u_eps = torch.matmul(eigenvector_T, eps_hat)
  #   u_t = torch.matmul(eigenvector_T, x)
  #   term1 = torch.matmul(eigenvector, (coeff1 * u_t))
  #   term2 = torch.matmul(eigenvector, (coeff2 * (u_t - (sigma_t[:, None, None] * u_eps))))
  #   mu_denoise = term1 + term2
    
  #   if adj:
  #     eps = gen_noise(mu_denoise, flags, sym = True)
  #   else: 
  #     eps = gen_noise(mu_denoise, flags, sym = False)

  #   return mu_denoise + torch.matmul(eigenvector, (torch.sqrt(sigma2_denoise) * eps))
    
  return denoise