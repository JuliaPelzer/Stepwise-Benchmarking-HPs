from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from torch.nn import MSELoss, L1Loss
import yaml

from code.train import init_data, init_model, infer
from code.metrics import LinfLoss, PATLoss

def evaluate(step:int, timedependent:bool):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data_train, loader_train = init_data(step, "train_split", timedependent)
    _, loader_val = init_data(step, "validation_split", timedependent)
    dataloaders = {"train": loader_train, "val": loader_val}
    try:
        _, loader_test = init_data(step, "test_split", timedependent)
        dataloaders["test"] = loader_test
    except:
        print("Test data not available, only evaluating on train and validation data")
    
    # --- model + load weights ---
    model = init_model(step, device, data_train)
    model_path = Path(f"step{step}.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")

    # --- evaluation loop ---
    measurements(step, device, dataloaders, model, timedependent)
    print(f"Evaluation for step {step} completed, metrics saved to metrics_step{step}.yaml")

def measurements(step:int, device, dataloaders:dict, model, timedependent:bool):
    metrics = {
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "timedependent": timedependent,
               }

    for case, loader in dataloaders.items():
        predictions, labels = infer(model, loader, device)

        if timedependent:
            # currently only evaluate last timestep, but maybe additionally evaluate all timesteps together?
            predictions = predictions[:, -1:, :, :]
            labels = labels[:, -1:, :, :]

        assert predictions.shape == labels.shape, f"Predictions and labels have different shapes: {predictions.shape} vs {labels.shape}"

        # eval only last timestep
        metrics[case] = metrics_one_timestep(predictions, labels)

        # visualization
        # visualize(predictions, labels, case, step) #if step=None: no pic saved, just shown

    yaml.safe_dump(metrics, open(f"metrics_step{step}.yaml", "w"))

def metrics_one_timestep(predictions:torch.Tensor, labels:torch.Tensor):
    pat_threshold = (0.1 - 0.8729731142642084) / (19.348517021154922 - 0.8729731142642084) # manually normed threshold of 0.1°C

    list_metrics = {
        "MSE (normed) [-]": MSELoss(),
        "RMSE (normed) [-]": None,
        "MAE (normed) [-]": L1Loss(),
        "Max Error (normed) [-]": LinfLoss(),
        "PAT 0.1 degC [%]": PATLoss(pat_threshold),
    }

    collected_metrics = {}
    for name, metric in list_metrics.items():
        if metric is not None:
            collected_metrics[name] = metric(predictions, labels).item()
    collected_metrics["RMSE (normed) [-]"] = torch.sqrt(MSELoss()(predictions, labels)).item()

    return collected_metrics

def aligned_cbar(*args, **kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes(
        "right", size=0.3, pad=0.05)
    cb = plt.colorbar(*args, cax=cax, **kwargs)
    return cb

def visualize(predictions:torch.Tensor, labels:torch.Tensor, case:str, step:int=None):
    for i, (prediction, label) in enumerate(zip(predictions, labels)):
        prediction = prediction.squeeze().numpy()
        label = label.squeeze().numpy()
        error = label - prediction

        plt.figure(figsize=(16, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(label.T, origin="lower", cmap="RdBu_r")#, vmin=0, vmax=1)
        plt.title("Label")
        plt.ylabel("y [cells]")
        plt.xlabel("x [cells]")
        aligned_cbar()

        plt.subplot(1, 3, 2)
        plt.imshow(prediction.T, origin="lower", cmap="RdBu_r")#, vmin=0, vmax=1)
        plt.title("Prediction")
        plt.xlabel("x [cells]")
        aligned_cbar()

        plt.subplot(1, 3, 3)
        plt.imshow(error.T, origin="lower", cmap="RdBu_r")
        plt.title("Error (GT - Prediction)")
        plt.xlabel("x [cells]")
        aligned_cbar()

        plt.tight_layout()
        if step is not None:
            visu_dir = Path(f"visus_step{step}")
            visu_dir.mkdir(exist_ok=True)
            plt.savefig(visu_dir / f"{case}_dp{i}.png")
            print(f"Saved visualization to {visu_dir / f'{case}_dp{i}.png'}")
        else:
            plt.show()