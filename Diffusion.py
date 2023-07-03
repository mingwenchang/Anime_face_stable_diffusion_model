import torch
import torch.nn as nn
from tqdm import tqdm

class Diffusion:
    """ This class implements the diffusion process described in the papers:
        https://arxiv.org/abs/2006.11239
        https://arxiv.org/abs/2102.09672?ref=assemblyai.com
    """

    def __init__(
            self,
            noise_steps: int = 1000,
            beta_1: float = 1e-4,
            beta_t: float = 0.02,
            img_size: int = 64,
            device: str = 'cpu') -> None:

        self.noise_steps = noise_steps  # T
        self.img_size = img_size
        self.device = device

        # Gaussian noise to the data according to a variance schedule β1, . . . , βT:
        self.beta = torch.linspace(beta_1, beta_t, noise_steps)  # shape: (noise_steps,)

        # alpha = 1 − βt
        self.alpha = 1.0 - self.beta  # shape: (noise_steps,)

        # alpha_bar = ∏t′=1,...,t (1 − αt′)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # shape: (noise_steps,)

        # Move to the target device:
        self.beta = self.beta.to(self.device)   # shape: (noise_steps,)
        self.alpha = self.alpha.to(self.device)  # shape: (noise_steps,)
        self.alpha_bar = self.alpha_bar.to(self.device)  # shape: (noise_steps,)

    def sample_timesteps(self, n: int = 1) -> torch.Tensor:
        """ Sample n timesteps uniformly from [1, noise_steps]"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).long().to(self.device)  # shape: (n,)

    def noise_image_at_time_t(self, x0: torch.Tensor, t: int or torch.Tensor) -> tuple[torch.Tensor]:
        """ Forward diffusion process:
        Add Gaussian noise to the image according to:

            xt(x0, epsilon) = √ᾱt x0 + √(1 − ᾱt) epsilon, where epsilon ~ N(0, I)

        Parameters:
            x0: image_to_be_noised, shape: (N, C, H, W)
            t: time step, shape: (N, )
        """
        if type(t) == int:
            # Convert t to a tensor:
            t = (torch.ones(x0.shape[0]) * t).long().to(self.device)   # shape: (N, )

        # Get the alpha_bar at time step t:
        alpha_bar_t = self.alpha_bar[t][:, None, None, None]  # shape: (N, 1, 1, 1)

        # Generate Gaussian noise:
        epsilon = torch.randn_like(x0)  # shape: (N, C, H, W); normal distribution with mean=0, std=1

        # Add Gaussian noise to the data:
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * epsilon  # shape: (N, C, H, W)
        return xt, epsilon

    def denoise_at_time_t(self, model: nn.Module, xt: torch.Tensor, t: int) -> torch.Tensor:
        """ Denoise at time step t

         xt-1 = 1/√αt * (xt - (1 - αt)/√(1 - ᾱt) * e(xt, t)) + √βt * z,

        where:
            xt-1 is the denoised image at time step t-1,
            xt is the noised image at time step t,
            z ~ N(0, I), and
            e(xt, t) is the predicted noise at time step t, x

        Parameters:
            model: the model used to predict the noise
            xt: the noised image at time step t, shape: (N, C, H, W)
            t: time step, shape: (N, )

        Returns:
            xt_1: the denoised image at time step t-1, shape: (N, C, H, W)
        """

        if t > 1:
            z = torch.randn(xt.shape).to(self.device)  # shape: (N, C, H, W)
        else:
            z = torch.zeros(xt.shape).to(self.device)  # shape: (N, C, H, W)

        # Convert t to a tensor:
        t = (torch.ones(xt.shape[0]) * t).long().to(self.device)   # shape: (N, )

        # Get the predicted noise:
        predicted_noise = model(xt, t)  # shape: (N, C, H, W)

        # Get the alpha at time step t:
        alpha_t = self.alpha[t][:, None, None, None]  # shape: (N, 1, 1, 1)

        # Get the alpha_bar at time step t:
        alpha_bar_t = self.alpha_bar[t][:, None, None, None]  # shape: (N, 1, 1, 1)

        # Get the beta at time step t:
        beta_t = self.beta[t][:, None, None, None]  # shape: (N, 1, 1, 1)

        # Denoise:
        xt_1 = 1 / torch.sqrt(alpha_t) * (
                xt - ((1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t))) * predicted_noise
        ) + torch.sqrt(beta_t) * z  # shape: (N, C, H, W)

        return xt_1  # x at time step t-1

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int) -> torch.Tensor:
        """ Sample n images from the model.
         This is the implementation of Algorithm 2 in the paper: https://arxiv.org/abs/2006.11239

        Parameters:
            model: the model used to predict the noise
            n: number of images to be sampled
         """
        # Set the model to evaluation mode:
        model.eval()

        # Sample noise: xT ~ N(0, I)
        x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)  # shape: (N, C, H, W)

        # Denoise from time step T to 1:
        for t in tqdm(reversed(range(1, self.noise_steps)), position=0):   # t = T, T-1, ..., 1
            x = self.denoise_at_time_t(model=model, xt=x, t=t)  # x at time step t-1

        # Clamp the values to [-1, 1]
        x0 = x  # x at time step 0

        # # Set the model back to training mode:
        model.train()
        return x0  # shape: (N, C, H, W)