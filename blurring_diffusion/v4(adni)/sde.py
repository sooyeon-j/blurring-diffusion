import abc
import torch
import numpy as np
import torch_dct as dct

class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.
    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.
    Useful for computing the log-likelihood via probability flow ODE.
    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.
    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)
    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.
    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # -------- Build the class for reverse-time SDE --------
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, feature, x, flags, t, is_adj=True):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t) if is_adj else sde_fn(feature, t)
        score = score_fn(feature, x, flags, t)
        drift = drift - diffusion[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # -------- Set the diffusion function to zero for ODEs. --------
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, feature, x, flags, t, is_adj=True):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t) if is_adj else discretize_fn(feature, t)
        score = score_fn(feature, x, flags, t)
        rev_f = f - G[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()


class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.
    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1
  
  @property
  def step(self):
    return self.N

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def frequency_scaling(self, eigenvalue, t, min_scale = 0.001, sigma_blur_max = 1):
    sigma_blur = sigma_blur_max * torch.sin(t * torch.pi / 2.) ** 2
    dissipation_time = sigma_blur ** 2. / 2
    
    # sigma_blur = torch.exp(torch.linspace(torch.log(torch.tensor(0.5)), torch.log(torch.tensor(20)), 1000)).to('cuda')
    # sigma_blur = sigma_blur[t][:, None, None]
    
    # freq = torch.pi * torch.linspace(0, max_node_num - 1, max_node_num) / max_node_num
    # labda = freq[:, None] ** 2 + freq[None, :] ** 2
   
    # labda = labda.to('cuda')
    scaling = torch.exp(-eigenvalue * dissipation_time[:, None, None]) * (1 - min_scale)
    return scaling #(bs, max_node_num, max_node_num)
    
  def noise_scheduling(self, t, logsnr_min = -10, logsnr_max = 10):
    limit_max = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max))) #(0, 1.5708], logsnr_max >= -21
    limit_min = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - limit_max #(-l.5708, 1.5708)
    logsnr = -2 * torch.log(torch.tan(limit_min * t + limit_max)) #tan grows expoentially as x goes to 1.57xx
    # logsnr = torch.tan(limit_min * t + limit_max) / 100 #max=-0.01, min=-10
    # logsnr_flip = torch.tan(limit_min * (self.T - t) + limit_max) / 100
    #import pdb; pdb.set_trace()
    return torch.sqrt(torch.sigmoid(logsnr)), torch.sqrt(torch.sigmoid(-logsnr)) #(bs,)
    #return logsnr_flip, logsnr
  
  def alpha_sigma(self, eigenvalue, t):
    freq_scaling = self.frequency_scaling(eigenvalue, t)
    # alpha = freq_scaling
    # sigma = 0.01
    a, sigma = self.noise_scheduling(t)
    alpha = a[:, None, None] * freq_scaling #(bs, max_node_num, max_node_num)
    return alpha, sigma
  
  def beta_t(self, x, t):
    freq_t = self.frequency_scaling(x.shape[-2], t)
    freq_t_minus_1 = self.frequency_scaling(x.shape[-2], (t - (1 / self.N)))
    # u_t = dct.idct_2d(torch.matmul(freq_t, dct.dct_2d(x, norm = 'ortho')), norm = 'ortho')
    # u_t_minus_1 = dct.idct_2d(torch.matmul(freq_t_minus_1, dct.dct_2d(x, norm = 'ortho')), norm = 'ortho')
    beta = 1 - (freq_t / freq_t_minus_1) ** 2
    
    return beta
  
  # -------- mean, std of the perturbation kernel --------
  def marginal_prob(self, x, eigenvalue, eigenvector, t):
    alpha, std = self.alpha_sigma(eigenvalue, t)
    # log_mean_coeff = -0.5 * t[:, None, None] * (1 - eigval)
    # mean = torch.matmul(torch.exp(log_mean_coeff), x)
    # std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    
    eigvec_t = eigenvector.transpose(-2, -1)
    x_freq = torch.matmul(eigvec_t, x)
    mean = torch.matmul(eigenvector, torch.matmul(alpha, x_freq))
    
    # std = torch.tensor(0.1).to('cuda')
    #std = torch.sqrt(1 - (freq_t ** 2))
    
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_sampling_sym(self, shape):
    x = torch.randn(*shape).triu(1) 
    return x + x.transpose(-1,-2)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None] * x - x
    G = sqrt_beta
    return f, G

  def transition(self, x, t, dt):
    # -------- negative timestep dt --------
    log_mean_coeff = 0.25 * dt * (2*self.beta_0 + (2*t + dt)*(self.beta_1 - self.beta_0) )
    mean = torch.exp(-log_mean_coeff[:, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std


class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.
    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape) 

  def prior_sampling_sym(self, shape):
    x = torch.randn(*shape).triu(1)
    x = x + x.transpose(-1,-2)
    return x 

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  def discretize(self, x, t):
    """SMLD(NCSN) discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    sigma = self.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas[timestep - 1].to(t.device))
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G

  def transition(self, x, t, dt):
    # -------- negative timestep dt --------
    std = torch.square(self.sigma_min * (self.sigma_max / self.sigma_min) ** t) - \
          torch.square(self.sigma_min * (self.sigma_max / self.sigma_min) ** (t + dt)) 
    std = torch.sqrt(std)
    mean = x
    return mean, std


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.
    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[:, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_sampling_sym(self, shape):
    x = torch.randn(*shape).triu(1) 
    return x + x.transpose(-1,-2)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
