from pathlib import Path
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from code.dataset import DatasetTimeResolved, DatasetLastTimestep

def train(step:int, timedependent:bool):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data_train, loader_train = init_data(step, "train_split", timedependent)
    _, loader_val = init_data(step, "validation_split", timedependent)
    
    model = init_model(step, device, data_train)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    n_epochs = 5
    
    best_val = float("inf")
    epochs = tqdm(range(n_epochs), desc="Training")
    for _ in epochs:
        model.train()
        train_loss = 0

        for batch in loader_train:
            x = batch["input"].to(device)
            y = batch["label"].to(device)

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
            torch.save(model.state_dict(), f"tmp_best_step{step}.pt")

        epochs.set_postfix_str(f"train loss: {train_loss:.2e}, val loss: {val_loss:.2e}")

    torch.save(model.state_dict(), f"step{step}.pt")
    Path(f"tmp_best_step{step}.pt").unlink(missing_ok=True)

def validate(model: torch.nn.Module, loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["label"].to(device)

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
            x = batch["input"].to(device)
            y = batch["label"].to(device)

            pred = model(x)
            pred_list.append(pred.detach().cpu())
            label_list.append(y.detach().cpu())
    return torch.cat(pred_list,dim=0), torch.cat(label_list,dim=0)

def init_data(step:int, case:str, timedependent:bool):
    if timedependent:
        data = DatasetTimeResolved(f"data/step{step}/{case}")
    else:
        data = DatasetLastTimestep(f"data/step{step}/{case}")

    shuffle = True if case == "train_split" else False
    dataloader = DataLoader(data, batch_size=4, shuffle=shuffle)

    print(f"Loaded {case}: {type(data).__name__} with {len(data)} samples of shapes {list(data[0]['input'].shape)} and {list(data[0]['label'].shape)} (inputs and labels), data points: {[name.stem for name in data.input_files]}")
    
    return data, dataloader

def init_model(step, device, data_train):
    if step == 1:
        from code.model import Step1 as Model
    elif step == 2:
        from code.model import Step2 as Model
    elif step == 3:
        from code.model import Step3 as Model
    elif step == "Dummy":
        from code.model import UNetDummy as Model

    model = Model(data_train.n_inputs(), data_train.n_outputs()).to(device)
    return model