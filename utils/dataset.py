import xarray as xr
from pathlib import Path
from torch import from_numpy, cat
from torch.utils.data import Dataset, ConcatDataset
from utils.config import DatasetConfig

class NinoData(Dataset):
    def __init__(self, path: Path, config: DatasetConfig):
        self.config = config
        self.seq_len = self.config.sequence_length
        self.variables = self.config.variables

        # Lazy loading
        ds = self._open_dataset(path)

        # Select the desired time, latitude, and longitude slices
        self._parse_slices(config)
        self.dataset = ds.sel(time = self.time_slice, lat = self.lat_slice, lon = self.lon_slice)
        
        assert list(ds.to_array(dim="variable").coords["variable"].values) == list(self.variables), "Variable order mismatch"
    
        # Preprocess and load the dataset into RAM
        self.tensor_data = self._preprocess(self.dataset)

    @staticmethod
    def _slice_from_cfg(s):
        if isinstance(s, dict):
            return slice(s["start"], s["stop"], s["step"])
        elif isinstance(s, slice):
            return s
        else:
            return slice(None)

    def _parse_slices(self, cfg: DatasetConfig) -> None:
        self.time_slice = self._slice_from_cfg(cfg.time_slice)
        self.lat_slice = self._slice_from_cfg(cfg.lat_slice)
        self.lon_slice = self._slice_from_cfg(cfg.lon_slice)
        
    def _open_dataset(self, path: Path) -> xr.Dataset:
        # Gets the xarray dataset
        data_arrays = self._load_netcdf_files(path, self.config.variables)
        try:
            ds = xr.merge(data_arrays, join='exact')
        except ValueError:
            ds = xr.merge(data_arrays, join='inner')
            print("Warning: Inner merge")
        return xr.merge([ds[var] for var in self.variables])
    
    def _preprocess(self, sliced_data: xr.Dataset) -> dict:
        # Compute data dependent attributes
        self.compute_means(sliced_data)
        self.compute_stds(sliced_data)
        self.compute_land_sea_mask(sliced_data)
        self.compute_length(sliced_data)

        # Standardize the data 
        standardized_data = self._standardize(sliced_data)

        # Transform the data to tensor
        if self.config.return_type == "dict":
            return {key: self._to_tensor(standardized_data[key]) for key in self.variables}
        else:
            return cat([self._to_tensor(standardized_data[key]) for key in self.variables], dim = 0)
    
    def _standardize(self, data: xr.Dataset) -> xr.Dataset:
        for key in self.config.variables:
            data[key] = (data[key] - self._means.sel(variable=key)) / self._stds.sel(variable=key)
        return data
    
    def compute_means(self, data: xr.Dataset) -> None:
        if self.config.stats is None:
            values = [data[key].mean().values for key in self.variables]
        else:
            values = [self.config.stats[key]['mean'] for key in self.variables]

        self._means = xr.DataArray(
            values, dims=["variable"], coords={"variable": self.variables}
        )
        self._means.name = "mean"

    def compute_stds(self, data: xr.Dataset) -> None:
        if self.config.stats is None:
            values = [data[key].std().values for key in self.variables]
        else:
            values = [self.config.stats[key]['std'] for key in self.variables]

        self._stds = xr.DataArray(
            values, dims=["variable"], coords={"variable": self.variables}
        )
        self._stds.name = "std"

    def compute_land_sea_mask(self, data: xr.Dataset) -> None:
        if "sftlf" in data:
            lsm = data["sftlf"]
        else:
            lsm = data[self.config.variables[0]].isel(time=0).isnull()
            lsm = lsm.drop_vars(["time", "month"], errors="ignore")
        self.land_sea_mask = self._to_tensor(lsm) # shape: [1, lat, lon]

    def compute_length(self, data: xr.Dataset) -> None:
        assert "time" in data.dims, "Dataset must have a time dimension"
        assert data.sizes["time"] > self.config.sequence_length, "Dataset must have more time steps than the sequence length"
        self._length = data.sizes["time"] - self.config.sequence_length
    
    @staticmethod
    def _to_tensor(data: xr.Dataset) :
        # Convert the data to a tensor
        data = data.fillna(0.0) #replace nan values with 0
        if isinstance(data, xr.Dataset):
            np_arr = data.to_array(dim='variable').values # add variable dimension in front
        elif isinstance(data, xr.DataArray):
            np_arr = data.values[None, :] # add empty variable dimension on front
        return from_numpy(np_arr).float().share_memory_() # share memory for multiprocessing
    
    @staticmethod
    def _load_netcdf_files(path: Path, variables: list = None) -> list:
        """
        Loads all .nc (netcdf) files, and filters to the given variables.
        Returns a list of xarray.DataArray objects.
        If variables is None or empty, all variables are returned.
        """
        files = list(sorted(Path(path).glob('*.nc')))
        if len(files) == 0:
            raise ValueError(f'No netcdf files found in {path}')

        data_arrays = []
        for file in files:
            da = xr.open_dataarray(file)
            if variables and da.name not in variables:
                continue
            if 'time' not in da.dims:
                raise ValueError('Dataset must have a time dimension')
            if 'lat' not in da.dims:
                raise ValueError('Dataset must have a lat dimension')
            if 'lon' not in da.dims:
                raise ValueError('Dataset must have a lon dimension')
            data_arrays.append(da)

        if variables and (len(data_arrays) != len(set(variables))):
            missing = set(variables) - {da.name for da in data_arrays}
            raise ValueError(f'Not all variables found. Missing: {missing}')

        return data_arrays
    
    # torch dataset methods
    def __len__(self):
        # Returns the length of the dataset
        return self._length

    def __getitem__(self, idx: int) -> dict:
        # Returns the data at the given index 
        if self.config.return_type == "dict":
            x = {key : self.tensor_data[key][:, idx: idx + self.seq_len] for key in self.variables}
        else:
            x = self.tensor_data[:, idx: idx + self.seq_len] # assumes [variable, time, ...]
        return x
    
class MultifileNinoDataset(ConcatDataset):
    # A dataset that combines multiple Nino datasets
    def __init__(self, path: Path, config: DatasetConfig, rank: int = None, world_size: int = None):
        self.config = config
        directories = self.get_directories(path)
        directories = self.shard_directories(directories, rank, world_size) if rank is not None else directories
        self.datasets = self.get_datasets(directories)
        super().__init__(self.datasets)
        self.land_sea_mask = self.datasets[0].land_sea_mask

    def shard_directories(self, directories: list, rank: int, world_size: int) -> list:
        # Split the directories into shards for each process
        assert rank is not None and world_size is not None, 'Rank and world size must be provided'
        assert rank < world_size, 'Rank must be less than world size'
        assert len(directories) >= world_size, 'Number of directories must be greater than or equal to world size'
        if len(directories) % world_size != 0:
            print('Warning: Number of directories is not divisible by world size. Some processes may have more directories than others.')
        return [directories[i] for i in range(rank, len(directories), world_size)]

    def get_datasets(self, directories: list) -> list:
        # Initialize a dataset for each directory
        assert len(directories) > 0, 'No directories found'
        return [NinoData(dir, self.config) for dir in directories]

    def get_directories(self, path: Path) -> list:
        # Get all directories in the root directory
        root_dir = Path(path) 
        assert root_dir.is_dir(), f'{root_dir} is not a directory'
        return [subdir for subdir in root_dir.iterdir() if subdir.is_dir()][:self.config.max_dirs]
