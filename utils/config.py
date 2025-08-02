from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence
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

@dataclass
class NetworkConfig:
    dim: int
    num_layers: Optional[int] = None
    num_tokens: Optional[int] = None
    num_latents: Optional[int] = None
    num_features: Optional[int] = None
    num_compute_blocks: Optional[int] = None
    num_cls: Optional[int] = None
    dim_in: Optional[int] = None
    dim_out: Optional[int] = None
    dim_noise: Optional[int] = None
    dim_heads: Optional[int] = 64
    dim_coords: Optional[int] = 32
    drop_prob: Optional[float] = 0.
    expansion_factor: Optional[int] = 2
    architecture: str = "vit"

@dataclass
class OptimConfig:
    batch_size: int = 64
    num_epochs: int = 1
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: Optional[float] = 0.01
    eta_min: Optional[float] = 1e-5

@dataclass
class DatasetConfig:
    sequence_length: int = 1
    variables: List[str] = field(default_factory=lambda: VARS)
    weights: Optional[list] = None
    time_slice: Optional[dict] = None
    lat_slice: Optional[dict] = None
    lon_slice: Optional[dict] = None
    stats: Optional[dict] = None
    frcst_tasks: Optional[dict] = None
    grid_size: dict = field(default_factory=lambda: {"lat": 64, "lon": 120})
    return_type: str = "tensor"
    eval_data: str = "picontrol"
    max_dirs: int = 100

@dataclass
class WorldConfig:
    mask_rates_src: dict = field(default_factory=lambda: {"mean": 0.0, "std": 1.0, "a": 0.0, "b": 1.0})
    mask_rates_tgt: dict = field(default_factory=lambda: {"mean": 0.0, "std": 1.0, "a": 0.0, "b": 1.0})
    var_rates_src: Optional[dict] = field(default_factory=lambda: {"mean": 0.0, "std": 1.0, "a": 0.0, "b": 1.0})
    var_rates_tgt: Optional[dict] = field(default_factory=lambda: {"mean": 0.0, "std": 1.0, "a": 0.0, "b": 1.0})
    tau: Optional[int] = None
    num_ens: Optional[int] = None
    alpha: Sequence = field(default_factory=lambda: [0.5, 1., 1.])
    num_tokens: int = 768
    patch_size: dict = field(default_factory=lambda: {"tt": 2, "hh": 4, "ww": 4})

@dataclass
class ModelConfig:
    decoder: Optional[NetworkConfig] = None
    modal: Optional[NetworkConfig] = None
    encoder: Optional[NetworkConfig] = None

@dataclass
class TrainerConfig:
    # Experiment settings
    seed: Optional[int] = 42
    job_name: Optional[str] = None # for resuming training
    stage1_id: Optional[str] = None # for loading pre-trained model
    epochs: int = 1

    # Logging
    use_wandb: bool = True
    wb_project: Optional[str] = None
    wb_tags: Optional[List[str]] = None

    # Trainer configs
    resume_training: bool = True
    val_loss_name: str = "loss"
    mixed_precision: bool = True
    use_ema: bool = False
    use_zero: bool = False
    use_compile: bool = False
    clip_gradients: bool = True
    scheduler_step: str = "epoch"
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
    model: ModelConfig
    optim: OptimConfig
    world: WorldConfig

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
            model = ModelConfig(
                decoder = NetworkConfig(**cfg.model.decoder) if exists(cfg.model.decoder) else None,
                encoder = NetworkConfig(**cfg.model.encoder) if exists(cfg.model.encoder) else None,
                modal = NetworkConfig(**cfg.model.modal) if exists(cfg.model.modal) else None
            ),
            optim =OptimConfig(**cfg.optim),
            world=WorldConfig(**cfg.world) 
        )