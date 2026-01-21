import torch
import einops
import torch_harmonics

from einops.layers.torch import EinMix
from utils.components import default, GatedFFN
from utils.config import WorldConfig


class SphericalDiffusionNoise(torch.nn.Module):
    def __init__(
            self,
            num_channels: int,
            num_steps: int = 36,
            num_lat: int = 180,
            num_lon: int = 360,
            sigma: float = 1.0,
            horizontal_length: float | list | torch.Tensor = 500., # correlation length w.r.t the earth radius
            temporal_length: float | list| torch.Tensor  = 1.0, # correlation length w.r.t dt = 1
            grid_type: str="equiangular",
            lat_slice: slice=None,
            lon_slice: slice=None,
        ):
        """
        A Random Field derived from a gaussian Diffusion Process on the sphere.
        For details see https://www.ecmwf.int/sites/default/files/elibrary/2009/11577-stochastic-parametrization-and-model-uncertainty.pdf,
        appendix 8.1.
        """
        super().__init__()
        #regional slices
        self.lat_slice = default(lat_slice, slice(0, num_lat, 1))
        self.lon_slice = default(lon_slice, slice(0, lon_slice, 1))

        # Number of latitudinal modes.
        self.nlat = num_lat
        self.nlon = num_lon
        self.nsteps = num_steps
        self.nchannels = num_channels

        # Inverse SHT
        self.isht = torch_harmonics.InverseRealSHT(self.nlat, self.nlon, grid=grid_type)
        self.lmax = self.isht.lmax
        self.mmax = self.isht.mmax      

        # make sure kT is a torch.Tensor of shape (nchannels, 1)
        horizontal_length = torch.as_tensor(horizontal_length)
        if horizontal_length.numel() == self.nchannels:
            horizontal_length = horizontal_length.reshape(self.nchannels, 1)
        elif horizontal_length.numel() == 1:
            horizontal_length = horizontal_length.expand(self.nchannels, 1)
        else:
            raise ValueError("horizontal_length must be a scalar or have length equal to nchannels")
        kT = 0.5 * (horizontal_length / 6378.0) ** 2 # scale = radius * sqrt(2 * kT)

        # same for lambd
        temporal_length = torch.as_tensor(temporal_length)
        if temporal_length.numel() == self.nchannels:
            temporal_length = temporal_length.reshape(self.nchannels, 1)
        elif temporal_length.numel() == 1:
            temporal_length = temporal_length.expand(self.nchannels, 1)
        else:
            raise ValueError("temporal_length must be a scalar or have length equal to nchannels")
        lambd = 1 / temporal_length.reshape(self.nchannels, 1) # lambda = dt / scale

        ls = torch.arange(self.lmax)
        ektllp1 = torch.exp(-kT * ls * (ls + 1))
        F0norm = torch.sum((2 * ls[1:] + 1) * ektllp1[..., 1:], dim=-1, keepdim=True)
        phi = torch.exp(-lambd)
        F0 = sigma * torch.sqrt(0.5 * (1 - phi**2) / F0norm)
        sigma_l = F0 * torch.exp(-0.5 * kT * ls * (ls + 1))
        # we multiply by 4 pi to get the correct variance. Check ECMWF docs and their Spherical Harmonic normalization
        sigma_l = torch.tensor(4 * torch.pi).sqrt() * sigma_l

        # we need the toeplitz matrix for the discounts:
        #            [    1,     0,   0, 0]
        # discount = [  phi,     1,   0, 0]
        #            [phi^2,   phi,   1, 0]
        #            [phi^3, phi^2, phi, 1]
        disc = phi.pow(torch.arange(self.nsteps))
        disc_flipped = disc.flip(dims = (1,)) # reversed order
        zero_padding = torch.zeros_like(disc)[:, 1:]
        flat = torch.cat([disc_flipped, zero_padding], dim = 1)
        toeplitz = flat.unfold(dimension = 1, size = self.nsteps, step = 1)
        discount = toeplitz.flip(dims = (1,)) # reversed strides

        # expand and register
        phi = einops.rearrange(phi, 'c 1 -> c 1 1 1 1')
        sigma_l = einops.rearrange(sigma_l, 'c l -> c 1 l 1 1')
        self.register_buffer("phi", phi, persistent=False)
        self.register_buffer("sigma_l", sigma_l, persistent=False)
        self.register_buffer("discount", discount, persistent=False)

    def forward(self, shape, rng = None):
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=False):
                # sample white noise
                eta_l = torch.empty(
                    (*shape, self.nchannels, self.nsteps, self.lmax, self.mmax, 2), 
                    dtype=torch.float32, device=self.phi.device
                    )
                # scale according to sigma
                state = self.sigma_l * eta_l.normal_(generator = rng)
                # the very first element in the time history requires a different weighting to sample the stationary distribution
                state[..., 0, :, :, :] = state[..., 0, :, :, :] / torch.sqrt(1.0 - self.phi[..., 0, :, :, :]**2)
                # get the right history by multiplying with the discount matrix
                state = torch.einsum("ctr, ...crlmu-> ...ctlmu", self.discount, state).contiguous()
                # inverse SHT
                cstate = torch.view_as_complex(state)
                eta = self.isht(cstate)
                # slice the region of interest
                eta = eta[..., self.lat_slice, self.lon_slice] 
        return eta
    
class RandomField(torch.nn.Module):
    def __init__(self, model_dim: int, world: WorldConfig, has_ffn: bool = True):
        super().__init__()
        horizontal = torch.tensor([500., 1000., 2000., 4000.])
        temporal = torch.tensor([1, 2, 4, 6])

        start = (180 // world.patch_sizes['hh'] - world.token_sizes['h']) // 2
        channels = len(horizontal) * len(temporal)

        self.noise_generator = SphericalDiffusionNoise(
            num_channels=channels,
            num_lat=180 // world.patch_sizes['hh'],
            num_lon=360 // world.patch_sizes['ww'],
            num_steps=world.token_sizes["t"],
            horizontal_length= einops.repeat(horizontal, 'h -> (h t)', h = len(horizontal), t = len(temporal)),
            temporal_length=einops.repeat(temporal, 't -> (h t)', h = len(horizontal), t = len(temporal)),
            lat_slice= slice(start, start + world.token_sizes['h']),
            lon_slice=slice(0, 2 * world.token_sizes['w'], 2)
        )

        self.projection = torch.nn.Sequential()
        
        self.projection.add_module(
            name = 'einmix', 
            module = EinMix(
                pattern = f'... c t h w -> ... {world.flat_token_pattern} d',
                weight_shape = 'v c d',
                c = channels, d = model_dim, **world.token_sizes,
            )
        )
        
        if has_ffn:
            self.projection.add_module(
                name = 'ffn',
                module = GatedFFN(model_dim)
            )
            

    def forward(self, tokens, rng = None):
        grf = self.noise_generator((tokens.size(0),), rng)
        return tokens, self.projection(grf)