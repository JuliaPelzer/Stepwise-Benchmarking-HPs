from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from torch.nn import MSELoss, L1Loss
from sklearn.metrics import mean_absolute_percentage_error as mape
import yaml

from code.train import init_data, init_model, infer
from code.metrics import LinfLoss, MaskedMAE, PATLoss, metric_shape_mismatch

def evaluate(step:int):
    """
    Currently, we evaluate only on the last 4 timesteps, as this is what would also be done in the competition.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # load data
    loader_train = init_data(step, "training_data", eval=True)
    dataloaders = {"train": loader_train}
    
    # load model
    model = init_model(step, device, loader_train)
    model_path = Path(f"results/step{step}_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")

    # visualize(model, dataloaders["val"], device, "val", step)

    # evaluate
    measurements(step, device, dataloaders, model)
    print(f"Evaluation for step {step} completed and saved to metrics_step{step}.yaml")

def measurements(step:int, device, dataloaders:dict, model):
    """
    Measure each split (train/val/test) individually, but avg over all seasons (4 last timesteps) and save yaml.
    Included metrics are:
    - model complexity:
        - n_params (number of trainable parameters)
        - model_size_MB (model size in MB (memory))
        ! need to ask participant for this number and trust it, but best models should be uploaded anyways for the purpose of open science, so any major misconduct is spotted anyways
        TODO how to communicate (through the kaggle interface)
    - classical ML metrics: 
        - MAE (L1)
        - MAPE (Mean Absolute Percentage Error)
        - MSE (L2)
        - RMSE (MSE^0.5)
        - Max Error (L_infty)
    - problem-specific metrics: 
        - shape mismatch: to judge if the plumes follow the correct paths
        - focused error (masked MAE 1.0°C): to only weight those regions that are relevant to the real-world application
        - [PAT0.1 (Percentage Above Threshold of 0.1°C deviation) : mask plume/no plume and substract with gt]
    - leaderboard:
        - focus on ML metrics + size metrics + 1-2 geoscience metrics
        - a * MSE + b * MAE + c * Max Error + d * shape_mismatch + e * focused error + f * log10(n_params)
        TODO define reasonable weights
    """


    metrics = {
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_size_MB": sum(p.element_size() * p.numel() for p in model.parameters() if p.requires_grad) / (1024 * 1024), # or just read weight-file-size
               }

    for case, loader in dataloaders.items():
        predictions, labels = infer(model, loader, device)
        assert predictions.shape == labels.shape, f"Predictions and labels have different shapes: {predictions.shape} vs {labels.shape}"

        metrics[case] = eval_metrics(predictions, labels, unnorm_fct=dataloaders[case].dataset.dataset.unnormalize)

    yaml.safe_dump(metrics, open(f"results/step{step}_metrics.yaml", "w"))

def eval_metrics(predictions:torch.Tensor, labels:torch.Tensor, unnorm_fct):
    pat0_1_threshold = 0.1
    pat1_threshold = 1.0
    T_init = 10.0 # initial temperature of 10°C

    list_metrics = {
        "MSE (normed) [-]": MSELoss(),
        "MSE (unnormed) [degC^2]": MSELoss(),
        "RMSE (normed) [-]": None,
        "RMSE (unnormed) [degC]": None,
        "MAE (normed) [-]": L1Loss(),
        "MAE (unnormed) [degC]": L1Loss(),
        "Max Error (normed) [-]": LinfLoss(),
        "Max Error (unnormed) [degC]": LinfLoss(), 
        "MAPE (normed) [%]": lambda preds, labels: mape(labels.cpu().numpy().reshape(-1), preds.cpu().numpy().reshape(-1)),
        "PAT 0.1degC (unnormed) [%]": PATLoss(pat0_1_threshold),
        "PAT 1.0degC (unnormed) [%]": PATLoss(pat1_threshold),
        "MaskedMAE (unnormed) [degC]": MaskedMAE(threshold=pat1_threshold, T_init=T_init),
        "Shape mismatch (unnormed) [-]": lambda preds, labels: metric_shape_mismatch(preds.cpu(), labels.cpu(), threshold=pat0_1_threshold, T_init=T_init),
    }

    unnormed_predictions = unnorm_fct(predictions, "labels")
    unnormed_labels = unnorm_fct(labels, "labels")

    collected_metrics = {}
    for name, metric in list_metrics.items():
        if "unnormed" in name:
            used_pred = unnormed_predictions
            used_lab = unnormed_labels
        else:
            used_pred = predictions
            used_lab = labels

        if metric is not None: # skip RMSE for now and calculate later
            if "MAPE" in name:
                collected_metrics[name] = metric(used_pred, used_lab)
            elif "Shape mismatch" in name or "MaskedMAE" in name:
                collected_metrics[name] = metric(used_pred, used_lab)[-1].item()
            else:
                collected_metrics[name] = metric(used_pred, used_lab).item()
            
    collected_metrics["RMSE (normed) [-]"] = torch.sqrt(MSELoss()(predictions, labels)).item()
    collected_metrics["RMSE (unnormed) [degC]"] = torch.sqrt(MSELoss()(unnormed_predictions, unnormed_labels)).item()

    return collected_metrics

def aligned_cbar(*args, **kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes(
        "right", size=0.3, pad=0.05)
    cb = plt.colorbar(*args, cax=cax, **kwargs)
    return cb

def visualize(model, dataloader, device, case:str, step:int=None):
    predictions, labels = infer(model, dataloader, device)
    for i, (prediction, label) in enumerate(zip(predictions, labels)):
        prediction = prediction.squeeze().numpy()
        label = label.squeeze().numpy()
        try:
            error = label - prediction
        except:
            error = torch.zeros((100,100))

        for j, (p_season, l_season, e_season) in enumerate(zip(prediction, label, error)):
            plt.figure(figsize=(16, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(l_season.T, origin="lower", cmap="RdBu_r")#, vmin=0, vmax=1)
            plt.title("Label")
            plt.ylabel("y [cells]")
            plt.xlabel("x [cells]")
            aligned_cbar()

            plt.subplot(1, 3, 2)
            plt.imshow(p_season.T, origin="lower", cmap="RdBu_r")#, vmin=0, vmax=1)
            plt.title("Prediction")
            plt.xlabel("x [cells]")
            aligned_cbar()

            plt.subplot(1, 3, 3)
            plt.imshow(e_season.T, origin="lower", cmap="RdBu_r")
            plt.title("Error (GT - Prediction)")
            plt.xlabel("x [cells]")
            aligned_cbar()

            plt.tight_layout()
            if step is not None:
                visu_dir = Path(f"results/step{step}_visus")
                visu_dir.mkdir(exist_ok=True)
                plt.savefig(visu_dir / f"{case}_dp{i}_season{j}.png")
                print(f"Saved visualization to {visu_dir / f'{case}_dp{i}_season{j}.png'}")
            else:
                plt.show()

        if i >= 2:
            break