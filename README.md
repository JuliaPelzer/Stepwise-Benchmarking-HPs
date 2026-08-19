Stepwise Benchmarking on Groundwater Heat Pump Modeling
=======================================================

This repository contains a very basic code framework to train and evaluate a model on the three separate steps of the `stepwise benchmark for groundwater heat pump modeling`. The code is structured in a way that allows you to easily modify the model architecture, training procedure, and input features. The data is already preprocessed; the dataloader contains a normalization function, so you can directly use it for training and evaluation. Of course all adaptations are welcome and encouraged.

## Hard facts about the data
- 3 steps with increasing complexity and less data
- spatial domain: 2D
- transient problem, so time series data BUT evaluated only on the last timestep (so not necessarily timestep-by-timestep prediction, but can also directly predict the last timestep if desired)

## Setup
### Install requirements
```bash
pip install -r requirements.txt
```

automatically installs the required packages that are not already installed.

### Download step-specific data
- the step-specific data is stored in 
    - step 1: https://doi.org/10.18419/DARUS-5806
    - step 2: https://doi.org/10.18419/DARUS-5807
    - step 3: https://doi.org/10.18419/DARUS-5808
    - step 3b: TODO! (add link, description, code)
- download and unzip the data files for one of the steps into `data/stepX` and replace `X` with the respective step number. 

**You can decide for one of three *training dataset* variants** (per step):
- `_steadystate`: labels contain only 4 timesteps - same as we test against (one per season, after 10 years of simulation time)
- `_timeseries`: labels contain all prior timesteps as additional information, potentially useful for training, i.e., 40 timesteps in total, but only the last 4 are used for evaluation
- `_interim_velocities`: same as steadystate, but additionally contains the steady-state velocity fields as interim labels, which can be used for training, but are not available for evaluation since they are outputs of the simulations rather than inputs

This code focuses on the steadystate dataset as a reference.

After the download into proper folders, the file structure should look like this:
```
data/
    stepX/
        general/               (collected information for normalization)
        training_data_VARIANT/
            Sim_a.npz           (contains inputs and labels)
            Sim_b.npz
            ...
        test_data/
            Sim_c.npz           (only contains inputs, not the labels, which are hidden for evaluation)
            ...
```

If you want to use more information from the raw simulation outputs, e.g., fixed subsurface parameters or other interim labels, please refer to https://doi.org/10.18419/DARUS-5920.

## Getting started
`python main.py` starts a dummy training and evaluation on a dummy dataset of just one training and one validation data points, for only 5 epochs to check that everything works.

The program has the additional parameters `step`, `mode` and `timedependent`, which can be set by running, e.g., `python main.py --step 1 --mode train --timedependent True`, with
- `step`: which of the datasets from the stepwise benchmark to use for training (1, 2, or 3)
- `mode`: whether to train a new model or evaluate an existing one (choices: "train", "eval")
- `timedependent`: whether to use the time-dependent dataset ("True") or the last-timestep dataset (default)

## YOUR adaptations
To modify the model, add your own model class into `code/model.py:StepX`. Using this name, it is automatically detected by the training script.
To modify the training procedure, adapt `train.py`.

You can also adapt the input field selection in a preprocessing step in `train.py:init_data`to use different input features or target variables. This code automatically norms the data to (0,1). To work with the unnormalized data or a different normalization, you can adapt the dataset in `dataset.py`.

Of course, you can also just download the data and use your own training framework.

## Testing your trained model against unseen data
If you would like to check your model's performance on the hidden test labels, you can upload its predictions to kaggle https://www.kaggle.com/organizations/sciml-geobench/. Be aware to keep the format the same as above to ensure that the evaluation works correctly. The evaluation metrics are the same as implemented here.

## Additional resources
The raw simulation data of all steps is available at https://doi.org/10.18419/DARUS-5920, which contains the full time series of not only every 3 months but every 5 days, unstructured data at a finer resolution around the wells, and h5 format. It additionally contains an easier dataset with no cooling during summer months, but only heating in winter, that we currently ignore.