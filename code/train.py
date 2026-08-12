from pathlib import Path
import yaml
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import numpy as np

from code.dataset import DatasetLast4Timesteps, DatasetNoLabels

def train(step:int):
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

    loader_train, loader_val = init_data(step, "training_data")
    
    model = init_model(step, device, loader_train)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    n_epochs = 100 #10_000 #5
    
    best_val = float("inf")
    epochs = tqdm(range(n_epochs), desc="Training")
    for _ in epochs:
        model.train()
        train_loss = 0

        for batch in loader_train:
            x = batch["inputs"].to(device)
            y = batch["labels"].to(device)

            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(loader_train)
        val_loss = validate(model, loader_val, loss_fn, device)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"results/tmp_best_step{step}.pt")

        epochs.set_postfix_str(f"train loss: {train_loss:.2e}, val loss: {val_loss:.2e}")

    torch.save(model.state_dict(), f"results/step{step}_model.pt")
    Path(f"results/tmp_best_step{step}.pt").unlink(missing_ok=True)

def validate(model: torch.nn.Module, loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["inputs"].to(device)
            y = batch["labels"].to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            total_loss += loss.item()

    return total_loss / len(loader)
    
def infer(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()

    pred_list = []
    label_list = []
    with torch.no_grad():
        for batch in loader:
            x = batch["inputs"].to(device)
            y = batch["labels"].to(device)

            pred = model(x)
            pred_list.append(pred.detach().cpu())
            label_list.append(y.detach().cpu())
    return torch.cat(pred_list,dim=0), torch.cat(label_list,dim=0)

def init_data(step:int, case:str):
    if case == "training_data":
        data = DatasetLast4Timesteps(f"data/step{step}/{case}")

        split_idx = int(0.8*len(data))
        # currently sorted: first 80% of data points are training, last 20% validation
        data_train = torch.utils.data.Subset(data, range(split_idx))
        data_val = torch.utils.data.Subset(data, range(split_idx, len(data)))
        loader_train = DataLoader(data_train, batch_size=64, shuffle=True)
        loader_val = DataLoader(data_val, batch_size=64, shuffle=False)
        print(f"Loaded {case}: {type(data).__name__} with {len(data)} samples of shapes inputs:{list(data[0]['inputs'].shape)} and labels:{list(data[0]['labels'].shape)}")
        return loader_train, loader_val
    else:
        data = DatasetNoLabels(f"data/step{step}/{case}")
        loader = DataLoader(data, batch_size=64, shuffle=False)
        print(f"Loaded {case}: {type(data).__name__} with {len(data)} samples of shapes inputs:{list(loader.dataset[0]['inputs'].shape)}")
        return loader

def init_model(step:int, device, dataloader: DataLoader, mode:str="train"):
    if step == 1:
        from code.model import Step1 as Model
    elif step == 2:
        from code.model import Step2 as Model
    elif step == 3:
        from code.model import Step3 as Model
    elif step == "Dummy":
        from code.model import UNetDummy as Model


    if mode in ["train", "eval"]:
        print(f"Initializing model for step {step} with input size {dataloader.dataset.dataset.n_inputs()} and output size {dataloader.dataset.dataset.n_outputs()} and prediction timesteps {dataloader.dataset.dataset.n_pred_timesteps()}")
        model = Model(dataloader.dataset.dataset.n_inputs(), dataloader.dataset.dataset.n_pred_timesteps()).to(device)
    elif mode == "bm": # benchmark mode
        print(f"Initializing model for step {step} with input size {dataloader.dataset.n_inputs()} and output size {dataloader.dataset.n_outputs()} and prediction timesteps {dataloader.dataset.n_pred_timesteps()}")
        model = Model(dataloader.dataset.n_inputs(), dataloader.dataset.n_pred_timesteps()).to(device)

    return model

def prep_for_competition(step:int):
    # preparation for benchmarking: only apply model to test data and save predictions as .npz, so that evaluation can be done without needing the model or code
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    loader = init_data(step, "test_data")
    model = init_model(step, device, loader, mode="bm")
    model_path = Path(f"results/step{step}_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path} for benchmarking preparation")

    # apply model to test data and save predictions
    preds_collected = []
    with torch.no_grad():
        for batch in loader:
            x = batch["inputs"].to(device)
            pred = model(x)
            pred = pred.cpu().numpy()
            preds_collected.append(pred)

    preds_collected = np.concatenate(preds_collected, axis=0)
    Path(f"results/step{step}_predictions").mkdir(parents=True, exist_ok=True)
    for pred, dp in zip(preds_collected, loader.dataset.dps):
        np.savez(f"results/step{step}_predictions/{dp.stem}.npz", prediction=pred)
        print(f"Saved prediction for data point {dp.stem} with shape {pred.shape}")
    trafo_npz_csv(Path(f"results/step{step}_predictions"), "prediction")

def trafo_npz_csv(diri, case:str):
    typei = "labels" if case=="labels" else "prediction"
    names = diri.glob("*.npz")

    # load example data to get shape information
    data = np.load(diri / next(names))[typei]
    d1 = np.arange(len(data[0].reshape(-1)))[np.newaxis,:]
    if case == "labels":
        d2 = np.full(len(data[0].reshape(-1)), "Public", dtype=object)[np.newaxis,:]
        header = ["ID", "Usage"]
        data_store = [d1, d2]
    else:
        header = ["ID"]
        data_store = [d1]

    names = diri.glob("*.npz") # reset generator, because previous next() call consumed one element
    for name in names:
        file = diri / name
        data = np.load(file)[typei]
        data_store.append(data.reshape(len(data),-1))
        header += [f"dp{file.stem.split('_')[1]}_c{i}" for i in range(data.shape[0])]
    data_store = np.concatenate(data_store, axis=0)
    data_store = data_store.transpose()
    print(data_store[:5,:])
    print(data_store.shape)
    print(header)
    np.savetxt(diri / f"combined_data_{case}.csv", data_store, delimiter=",", fmt="%s", header=",".join(header), comments="")

    # store format information in yaml
    with open(diri / "orig_shape.yaml", "w") as f:
        yaml.dump({"shape": list(data[0].shape)}, f)

if __name__ == "__main__":
    diri = Path("/scratch/sgs/pelzerja/datasets/feflow/lorentz/data/step1/test_hidden_labels")
    trafo_npz_csv(diri, "labels")
    # diri = Path("/scratch/sgs/pelzerja/datasets/feflow/lorentz/results/step1_predictions")
    # trafo_npz_csv(diri, "prediction")