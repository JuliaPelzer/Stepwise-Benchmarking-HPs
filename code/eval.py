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
    loader_train, loader_val = init_data(step, "training_data", data_type="eval")
    dataloaders = {"train": loader_train, "val": loader_val}
    
    # load model
    model = init_model(step, device, loader_train)
    model_path = Path(f"results/step{step}_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")

    # evaluate
    measurements(step, device, dataloaders, model)
    print(f"Evaluation for step {step} completed and saved to metrics_step{step}.yaml")

def measurements(step:int, device, dataloaders:dict, model):
    """
    Measure each split (train/val/test) individually, but avg over all seasons (4 last timesteps) and save yaml.
    TODO Clean up this header
    Included metrics are:
    - model complexity:
        - n_params (number of trainable parameters) (if possible?)
        - model_size_MB (model size in MB (memory)) (if possible?) TODO - need to ask participant for this number and trust it... TODO how to communicate
    - classical ML metrics: 
        - MSE (L2)
        - RMSE (MSE^0.5)
        - MAE (L1)
        - Max Error (L_infty)
        - MAPE (Mean Absolute Percentage Error)
        - [SSIM]
    - problem-specific metrics: 
        - PAT1.0 (Percentage Above Threshold of 1.0°C deviation)
        - PAT0.1 (Percentage Above Threshold of 0.1°C deviation) : mask plume/no plume and substract with gt -> judge shape (achtung chaotic) TODO: compare to No. cells or No. cells GT with plume (>0.1°C)? TODO PAT correct implemented?
        -> naa - better use shape mismatch metric (see below) TODO check if normed or unnormed
        - masked MAE 1.0°C : to only weight those regions that are relevant to the real-world application
        - [Wasserstein distance]
        - [KGE (Kling-Gupta efficiency)] TODO @FB
	    - [connectivity with all 4 seasons summed up/ max] NÖ

    - where to evaluate? TODO
        - CURRENTLY: everywhere
        - alternative idea: everywhere except for around heat pump wells (because there nothing new would be installed anyways)
        - at observation points (extraction wells) only, because that's where it's relevant

    - leaderboard:
        - focus on ML metrics + size metrics + 1-2 geoscience metrics
        - a * MSE + b * MAE + c * Max Error + d * PAT1.0 + e * PAT0.1 + f * log10(n_params)
        TODO figure out weights
    """


    metrics = {
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_size_MB": sum(p.element_size() * p.numel() for p in model.parameters() if p.requires_grad) / (1024 * 1024), # TODO CHECK!! or else just read weight-file-size
               }

    for case, loader in dataloaders.items():
        predictions, labels = infer(model, loader, device)
        assert predictions.shape == labels.shape, f"Predictions and labels have different shapes: {predictions.shape} vs {labels.shape}"

        metrics[case] = eval_metrics(predictions, labels, unnorm_fct=dataloaders[case].dataset.dataset.unnormalize)

        # visualization
        # visualize(predictions, labels, case, step) #if step=None: no pic saved, just shown

    yaml.safe_dump(metrics, open(f"results/step{step}_metrics.yaml", "w"))

def eval_metrics(predictions:torch.Tensor, labels:torch.Tensor, unnorm_fct):
    quick_norm = lambda x: (x - 5.0) / (15.0 - 5.0) # manually normed to (0,1) for 5-15°C
    pat0_1_threshold = 0.1
    pat0_1_threshold_normed = quick_norm(pat0_1_threshold) # manually normed threshold of 0.1°C
    pat1_threshold = 1.0
    pat1_threshold_normed = quick_norm(pat1_threshold) # manually normed threshold of 1.0°C
    T_init = 10.0 # initial temperature of 10°C
    T_init_normed = (T_init - 5.0) / (15.0 - 5.0) # manually normed initial temperature of 10°C

    list_metrics = {
        "MSE (normed) [-]": MSELoss(),
        "RMSE (normed) [-]": None,
        "MAE (normed) [-]": L1Loss(),
        "Max Error (normed) [-]": LinfLoss(),
        "MAPE (normed) [%]": lambda preds, labels: mape(labels.cpu().numpy().reshape(-1), preds.cpu().numpy().reshape(-1)),
        "PAT 0.1 degC [%]": PATLoss(pat0_1_threshold_normed),
        "Shape mismatch (normed) [-]": lambda preds, labels: metric_shape_mismatch(preds.cpu(), labels.cpu(), threshold=pat0_1_threshold, T_init=T_init, unnorm=unnorm_fct),
        # "PAT 1.0 degC [%]": PATLoss(pat1_threshold or _normed),
        "MaskedMAE (normed) [-]": MaskedMAE(threshold=pat1_threshold_normed, T_init=T_init_normed),
    }

    collected_metrics = {}
    for name, metric in list_metrics.items():
        if metric is not None:
            if "MAPE" in name:
                collected_metrics[name] = metric(predictions, labels)
            elif "Shape mismatch" in name:
                collected_metrics[name] = metric(predictions, labels)[-1].item()
            elif "MaskedMAE" in name:
                collected_metrics[name] = metric(predictions, labels)[-1].item()
            else:
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
            visu_dir = Path(f"results/step{step}_visus")
            visu_dir.mkdir(exist_ok=True)
            plt.savefig(visu_dir / f"{case}_dp{i}.png")
            print(f"Saved visualization to {visu_dir / f'{case}_dp{i}.png'}")
        else:
            plt.show()