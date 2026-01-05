from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import OmegaConf
from math import prod

OmegaConf.register_new_resolver(
    "prod",
    lambda *args: prod(args),
    replace=True,
)

OmegaConf.register_new_resolver(
    "len",
    lambda x: len(x),
    replace=True,
)

OmegaConf.register_new_resolver(
    "div",
    lambda a,b: a/b,
    replace=True,
)

OmegaConf.register_new_resolver(
    "div_comp",
    lambda x, y: (i // j for i, j in zip(x, y)),
    replace=True,
)

def exists(x):
    return x is not None

def default(x, default_value):
    return x if exists(x) else default_value

DATA_DIR = Path("/mnt/lustre/work/ludwig/jthuemmel54/data/")

VARS = ['temp_ocn_0a', 'temp_ocn_1a', 'temp_ocn_3a', 'temp_ocn_5a', 'temp_ocn_8a', 'temp_ocn_11a', 'temp_ocn_14a', 'tauxa', 'tauya']

LENS_STATS = {
        'temp_ocn_0a': {'mean': -1.460749187956891e-17, 'std': 0.6716897243585809},
        'temp_ocn_1a': {'mean': 6.588296232102237e-18, 'std': 0.6718217430638053},
        'temp_ocn_3a': {'mean': 6.0490856156816606e-18, 'std': 0.7230672257454342},
        'temp_ocn_5a': {'mean': -1.4047735896765925e-17, 'std': 0.8418841484002196},
        'temp_ocn_8a': {'mean': -1.89722131875779e-18, 'std': 0.9127529247324552},
        'temp_ocn_11a': {'mean': 4.809813862225761e-19, 'std': 0.9509003051051766},
        'temp_ocn_14a': {'mean': -1.1246988115251004e-17, 'std': 0.9060856579963922},
        'tauxa': {'mean': -3.9094611618755137e-20, 'std': 0.19904808837784083},
        'tauya': {'mean': 1.7936438524139273e-20, 'std': 0.1323912433010137},
        }

PICONTROL_STATS = {
                'temp_ocn_0a': {'mean': 5.32304118e-18, 'std': 0.63290073}, 
                'temp_ocn_1a': {'mean': 1.02516498e-17, 'std': 0.63289313}, 
                'temp_ocn_3a': {'mean': 7.99551042e-18, 'std': 0.68246464}, 
                'temp_ocn_5a': {'mean': 1.66074617e-18, 'std': 0.80192137}, 
                'temp_ocn_8a': {'mean': -6.57904015e-18, 'std': 0.87488439}, 
                'temp_ocn_11a': {'mean': -4.43419892e-18, 'std': 0.90941989}, 
                'temp_ocn_14a': {'mean': -5.75602839e-18, 'std': 0.86523064}, 
                'tauxa': {'mean': -4.03492854e-19, 'std': 0.19840702}, 
                'tauya': {'mean': 1.71288663e-19, 'std': 0.13144593}
            }

GODAS_STATS = {'temp_ocn_0a': {'mean': -2.2515301306437936e-16, 'std': 0.5806380769447685}, 
                'temp_ocn_1a': {'mean': -6.582094131922381e-17, 'std': 0.6208909573494896}, 
                'temp_ocn_3a': {'mean': -4.743647150247509e-17, 'std': 0.8264447254204249}, 
                'temp_ocn_5a': {'mean': -2.235642317221912e-17, 'std': 1.0546172785963273}, 
                'temp_ocn_8a': {'mean': -8.477297408057756e-17, 'std': 1.2049644188990674}, 
                'temp_ocn_11a': {'mean': 3.353351510613191e-16, 'std': 1.1811549404412836}, 
                'temp_ocn_14a': {'mean': -4.672355811870539e-16, 'std': 1.062448117706782}, 
                'tauxa': {'mean': 7.085307169190875e-06, 'std': 0.02265642542041636}, 
                'tauya': {'mean': -7.363466657744754e-06, 'std': 0.015571367609252504}}

@dataclass
class NetworkConfig:
    dim: int
    num_layers: Optional[int] = None
    num_latents: Optional[int] = None
    num_compute_blocks: Optional[int] = None
    num_cls: Optional[int] = None
    num_tails: Optional[int] = None
    dim_in: Optional[int] = None
    dim_out: Optional[int] = None
    dim_noise: Optional[int] = None
    dim_heads: Optional[int] = 64
    dim_coords: Optional[int] = 32
    drop_path: Optional[float] = 0.0
    expansion_factor: Optional[int] = 2
    wavelengths: Optional[List] = None
    use_checkpoint: Optional[bool] = True
    backbone: Optional[str] = None

@dataclass
class DatasetConfig:
    sequence_length: int = 1
    variables: List[str] = field(default_factory=lambda:VARS)
    eval_variables: List[str] = field(default_factory=lambda:VARS)
    time_slice: Optional[dict] = None
    lat_slice: Optional[dict] = None
    lon_slice: Optional[dict] = None
    stats: Optional[dict] = field(default_factory=lambda:LENS_STATS)
    return_type: str = "tensor"
    eval_data: str = "picontrol"
    max_dirs: int = 100

@dataclass
class ObjectiveConfig:
    # prior
    prior: str = 'dirichlet'
    # frcst context
    tau: int = 2
    # args for masked diffusion sampling
    stratify: bool = True
    progressive: bool = False
    discretise: bool = False
    # event dims for prior
    event_dims: List[str] = field(default_factory=lambda: ['t'])
    # kumaraswamy concentrations
    c1: float = 1.
    c0: float = 1.
    # dirichlet concentration
    alpha: float = 0.5
    # multinomial bounds
    k_min: Optional[int] = None
    k_max: Optional[int] = None
    # schedule bounded [eps, 1-eps]
    epsilon: float = 0.05

@dataclass
class WorldConfig:
    field_sizes: dict
    patch_sizes: dict
    batch_size: int

    # derived fields
    field_layout: tuple = field(init=False)
    patch_layout: tuple = field(init=False)
    token_sizes: dict = field(init=False)
    token_shape: tuple = field(init=False)
    
    num_tokens: int = field(init=False)
    num_elements: int = field(init=False)
    dim_tokens: int = field(init=False)

    field_pattern: str = field(init=False)
    token_pattern: str = field(init=False)
    patch_pattern: str = field(init=False)
    flat_token_pattern: str = field(init=False)
    flat_patch_pattern: str = field(init=False)
    flatland_pattern: str = field(init=False)

    def __post_init__(self):
        self.field_layout = tuple(self.field_sizes.keys())
        self.patch_layout = tuple(self.patch_sizes.keys())

        self.token_sizes = {
            ax: (self.field_sizes[ax] // self.patch_sizes[f'{ax*2}'])
            for ax in self.field_layout
        }
        self.token_shape = tuple(self.token_sizes[ax] for ax in self.field_layout)
        
        self.num_tokens = prod(self.token_sizes.values())
        self.num_elements = prod(self.field_sizes.values())
        self.dim_tokens = prod(self.patch_sizes.values())

        field = " ".join(f"({f} {p})" for f, p in zip(self.field_layout, self.patch_layout))
        self.field_pattern = f"b {field}"
        self.patch_pattern = ' '.join(self.patch_layout)
        self.token_pattern = ' '.join(self.field_layout)
        
        self.flat_token_pattern = f"({self.token_pattern})"
        self.flat_patch_pattern = f"({self.patch_pattern})"
        self.flatland_pattern = f"b {self.flat_token_pattern} {self.flat_patch_pattern}"

@dataclass
class TrainerConfig:
    # Experiment settings
    seed: Optional[int] = 42
    job_name: Optional[str] = None # for resuming training
    stage1_id: Optional[str] = None # for loading pre-trained model
    resume_training: bool = True
    val_loss_name: str = "loss"
    
    # Logging
    use_wandb: bool = True
    wb_project: Optional[str] = None
    wb_tags: Optional[List[str]] = None
    save_eval: bool = False

    # Optimization
    epochs: Optional[int] = 1

    use_zero: bool = False
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.01

    schedulers: Optional[list ] = None
    scheduler_step: str = "batch"
    
    use_ema: bool = False
    ema_decay: float = 0.9999
    
    spectral_loss_weight: float = 0.
    mixed_precision: bool = True
    clip_gradients: bool = True
    clip_value: float = 1.
    
    num_workers: int = 4

    # Paths
    picontrol_path: str | Path = field(default_factory=lambda: DATA_DIR / Path("CMIP6_LENS/CESM2/piControl/temp_ocean_1_2_grid/processed"))
    lens_path: str | Path = field(default_factory=lambda: DATA_DIR / Path("CMIP6_LENS/CESM2/historical_levels/temp_ocean/1_2_grid/all_ensembles"))
    godas_path: str | Path = field(default_factory=lambda: DATA_DIR / Path("enso_data_pacific/godas/temp_ocean/1_2_grid"))
    oras5_path: Optional[str] | Optional[Path] = '/mnt/lustre/work/ludwig/jthuemmel54/data/enso_data_pacific/oras5/temp_ocean/1_2_grid'
    model_dir: str = "/mnt/lustre/work/ludwig/jthuemmel54/recurrence/runs/"
    
    
@dataclass
class MTMConfig:
    trainer: TrainerConfig
    data: DatasetConfig
    model: NetworkConfig
    world: WorldConfig
    objective: ObjectiveConfig

    @classmethod
    def from_omegaconf(cls, cfg: dict | OmegaConf):
        # Check if cfg is already an OmegaConf object
        if not isinstance(cfg, OmegaConf):
            # Convert dict to OmegaConf
            cfg = OmegaConf.create(cfg)
        # Convert OmegaConf to dataclass

        return cls(
            trainer=TrainerConfig(**cfg.trainer),
            data=DatasetConfig(**cfg.data),
            model =  NetworkConfig(**cfg.model),
            world=WorldConfig(**cfg.world),
            objective=ObjectiveConfig(**cfg.objective)
        )