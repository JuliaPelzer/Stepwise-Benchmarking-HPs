from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import yaml

class DatasetTimeResolved(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.inputs_path = self.root_dir / "inputs_unnormed" #TODO check dir name in DaRUS/codabench
        self.labels_path = self.root_dir / "labels_unnormed"
        self.info_path = self.root_dir.parent / "general" / "properties_info_normalization.yaml"
        assert self.inputs_path.exists() and self.labels_path.exists() and self.info_path.exists(), f"Input, label, or info path does not exist for {root_dir}"
        
        self.input_files = sorted(list(self.inputs_path.glob("*.npz")))
        self.label_files = sorted(list(self.labels_path.glob("*.npz")))

        if len(self.input_files) == 0:
            raise FileNotFoundError(f"No .npz files found in {self.inputs_path}")
        assert len(self.input_files) == len(self.label_files), "Mismatch in number of input and label files"

        self.info = yaml.safe_load(self.info_path.open())
        self.inputs_mins = np.array([item['min'] for item in sorted(self.info['Inputs'].values(), key=lambda x: x['index'])])[:,None, None]
        self.inputs_maxs = np.array([item['max'] for item in sorted(self.info['Inputs'].values(), key=lambda x: x['index'])])[:,None, None]
        self.labels_mins = np.array([item['min'] for item in sorted(self.info['Labels'].values(), key=lambda x: x['index'])])[:,None, None]
        self.labels_maxs = np.array([item['max'] for item in sorted(self.info['Labels'].values(), key=lambda x: x['index'])])[:,None, None]

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, i):
        input_data = np.load(self.input_files[i])["inputs"]
        label_data = np.load(self.label_files[i])["labels"]
            
        # Convert to tensor and ensure float type
        input_tensor = torch.from_numpy(input_data).float()
        label_tensor = torch.from_numpy(label_data).float()

        # norm data to (0,1)
        input_tensor = (input_tensor - torch.tensor(self.inputs_mins)) / (torch.tensor(self.inputs_maxs) - torch.tensor(self.inputs_mins))
        label_tensor = (label_tensor - torch.tensor(self.labels_mins)) / (torch.tensor(self.labels_maxs) - torch.tensor(self.labels_mins))
        
        return {"input": input_tensor.to(torch.float32), "label": label_tensor.to(torch.float32)}

    def n_inputs(self):
        return len(self.info["Inputs"])
    
    def n_outputs(self):
        return len(self.info["Labels"])
    
    def n_timesteps(self):
        # Assuming time is the first dimension of the input data
        sample_input = np.load(self.input_files[0])["inputs"]
        return sample_input.shape[0]

class DatasetLastTimestep(DatasetTimeResolved):
    # TODO
    def __init__(self, root_dir):
        super().__init__(root_dir)

    def __getitem__(self, i):
        input_data = np.load(self.input_files[i])["inputs"]
        # Select only the last timestep (assuming time is the first dimension)
        label_data = np.load(self.label_files[i])["labels"][-1:,]

        # Convert to tensor and ensure float type
        input_tensor = torch.from_numpy(input_data).float()
        label_tensor = torch.from_numpy(label_data).float()

        # norm data to (0,1)
        input_tensor = (input_tensor - torch.tensor(self.inputs_mins)) / (torch.tensor(self.inputs_maxs) - torch.tensor(self.inputs_mins))
        label_tensor = (label_tensor - torch.tensor(self.labels_mins)) / (torch.tensor(self.labels_maxs) - torch.tensor(self.labels_mins))

        return {"input": input_tensor.to(torch.float32), "label": label_tensor.to(torch.float32)}
    
    def n_timesteps(self):
        return 1