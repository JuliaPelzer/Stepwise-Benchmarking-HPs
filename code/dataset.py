from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import yaml

class DatasetLast4Timesteps(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        assert self.root_dir.exists(), f"Data directory {self.root_dir} does not exist - please download the data first and check that your data is in training_data, not in training_data_timeseries or training_data_steadystate or training_data_interim_velocities"
        self.info_path = self.root_dir.parent / "general" / "normalization_info.yaml"
        self.dps = sorted(list(self.root_dir.glob("*.npz")))
        assert len(self.dps) > 0 and self.info_path.exists(), f"Data or normalization info does not exist for {self.root_dir}"

        self.info = yaml.safe_load(self.info_path.open())
        self.inputs_mins, self.inputs_maxs, self.labels_mins, self.labels_maxs = self.get_min_max_from_yaml()

        # # for checking
        # test_dp = self[0]
        # print(f"Dataset initialized with {len(self.dps)} data points. Sample input shape: {test_dp['inputs'].shape}, Sample label shape: {test_dp['labels'].shape}")

    def __len__(self):
        return len(self.dps)

    def __getitem__(self, i):
        dp = self.dps[i]
        data = dict(np.load(dp))

        input_data = data['inputs']
        input_tensor = torch.from_numpy(input_data)

        # norm data to (0,1)
        input_tensor = self.normalize(input_tensor, "inputs")
        try:
            label_data = data['labels'][-4:,] # Select only the last 4 timesteps (assuming time is the first dimension). Works for both the steadystate and the timedependent cases, as the steadystate case has only 4 timesteps anyway.
            label_tensor = torch.from_numpy(label_data)
            label_tensor = self.normalize(label_tensor, "labels")
            return {"inputs": input_tensor.to(torch.float32), "labels": label_tensor.to(torch.float32)}

        except KeyError: # in case labels are missing, e.g. for test data (on purpose)
            return {"inputs": input_tensor.to(torch.float32)}

    def normalize(self, data, case):
        if case == "inputs":
            mins = self.inputs_mins
            maxs = self.inputs_maxs
        elif case == "labels":
            mins = self.labels_mins
            maxs = self.labels_maxs
        else:
            raise ValueError("Invalid case. Choose 'inputs' or 'labels'.")

        data = (data - torch.tensor(mins)) / (torch.tensor(maxs) - torch.tensor(mins))
        return data

    def unnormalize(self, data, case):
        if case == "inputs":
            mins = self.inputs_mins
            maxs = self.inputs_maxs
        elif case == "labels":
            mins = self.labels_mins
            maxs = self.labels_maxs
        else:
            raise ValueError("Invalid case. Choose 'inputs' or 'labels'.")

        data = data * (torch.tensor(maxs) - torch.tensor(mins)) + torch.tensor(mins)
        return data

    def n_inputs(self):
        return len(self.info["Inputs"])
    
    def n_outputs(self):
        return len(self.info["Labels"])
    
    def n_pred_timesteps(self):
        return 4
    
    def get_min_max_from_yaml(self):
        inputs_mins = np.array([item['min'] for item in sorted(self.info['Inputs'].values(), key=lambda x: x['index'])])[:,None, None]
        inputs_maxs = np.array([item['max'] for item in sorted(self.info['Inputs'].values(), key=lambda x: x['index'])])[:,None, None]
        labels_mins = np.array([item['min'] for item in sorted(self.info['Labels'].values(), key=lambda x: x['index'])])[:,None, None]
        labels_maxs = np.array([item['max'] for item in sorted(self.info['Labels'].values(), key=lambda x: x['index'])])[:,None, None]
        return inputs_mins, inputs_maxs, labels_mins, labels_maxs
    
class DatasetNoLabels(DatasetLast4Timesteps):
    def __init__(self, root_dir):
        super().__init__(root_dir)

    def __getitem__(self, i):
        dp = self.dps[i]
        data = dict(np.load(dp))

        input_data = data['inputs']
        input_tensor = torch.from_numpy(input_data)

        # norm data to (0,1)
        input_tensor = self.normalize(input_tensor, "inputs")
        return {"inputs": input_tensor.to(torch.float32)}