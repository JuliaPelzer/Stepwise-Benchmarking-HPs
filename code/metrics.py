import torch.nn as nn
import torch

class LinfLoss(nn.Module):
    def __init__(self):
        super(LinfLoss, self).__init__()

    def forward(self, output, target):
        return torch.max(torch.abs(output - target))

class PATLoss(nn.Module):
    """
    Percentage above Threshold, unit [%]
    Goal: this should be as small as possible, ideally 0%.

    It counts the number of cells that have an absolute error above a certain threshold, divided by the total number of cells, * 100 to get a percentage.
    """

    def __init__(self, pat_threshold: float):
        super(PATLoss, self).__init__()
        self.pat_threshold = pat_threshold

    def forward(self, output, label):
        pat = torch.sum(torch.abs(output - label) > self.pat_threshold) / torch.numel(output)
        return pat * 100
    
class MaskedMAE(nn.Module):
    """
    Normed Masked Mean Absolute Error (MAE) loss function.
    Computes the MAE only for the elements of the output and label tensors that exceed a specified threshold in the label tensor.
    Normed according to the mean absolute value of the masked label tensor.
    This is useful for focusing the loss calculation on specific regions of interest in the data.
    """

    def __init__(self, threshold: float = 0.0, T_init: float = 0.0):
        super(MaskedMAE, self).__init__()
        self.threshold = threshold
        self.T_init = T_init

    def forward(self, prediction, label):
        # set all values within the threshold to zero, 
        # and all outside of it, if they are positive to value-T_init-threshold, if they are negative to value-T_init+threshold
        # and compute the mean absolute error of the remaining values
        label_masked = torch.where(label>self.T_init+self.threshold, label-self.T_init-self.threshold, 0) + torch.where(label<self.T_init-self.threshold, label-self.T_init+self.threshold, 0)
        prediction_masked = torch.where(prediction>self.T_init+self.threshold, prediction-self.T_init-self.threshold, 0) + torch.where(prediction<self.T_init-self.threshold, prediction-self.T_init+self.threshold, 0)
        
        assert torch.mean(torch.abs(label_masked)) != 0, "Denominator is zero"
        if label_masked.ndim == 3:
            nominator = torch.mean(torch.abs(label_masked - prediction_masked))
            denominator = torch.mean(torch.abs(label_masked))
        elif label_masked.ndim == 4:
            nominator = torch.mean(torch.abs(label_masked - prediction_masked), dim=(1,2,3)) 
            denominator = torch.mean(torch.abs(label_masked), dim=(1,2,3))
        else:
            raise ValueError("MaskedMAE only supports 3D or 4D tensors, but got tensor with shape: {}".format(label_masked.shape))
        
        valid = denominator != 0 # do I want to handle zero-cases differently than excluding them?
        normed_mae = nominator[valid] / denominator[valid]
        normed_mae = torch.mean(normed_mae) # average over batch
        return label_masked, prediction_masked, normed_mae

def metric_shape_mismatch(prediction, label, threshold:float=0.1, T_init:float=10.0):
    """
    Computes the shape match metric between a label and a prediction.
    
    Args:
        label (np.ndarray): The ground truth label.
        prediction (np.ndarray): The predicted output.
        threshold (float): The threshold to binarize the prediction.
    """
    # we expect the data to not be normalized
    assert (prediction.max() > 1.5 and prediction.min() > 0.5) and (label.max() > 1.5 and label.min() > 0.5), "either labels or predictions are not UNnormed"
    assert label.shape == prediction.shape, "Label and prediction must have the same shape."
    assert label.shape[-3] >= 3, "Label and prediction must have at least 4 channels, i.e., timesteps."

    def inner(label, prediction):
        preds = torch.where(torch.abs(prediction - T_init) > threshold, 1, 0)
        pred = torch.sum(preds, dim=0) # sum over batch and timesteps, so that we get a 2D array of shape (height, width)
        pred = torch.where(pred>0, 1, 0)

        labs = torch.where(torch.abs(label - T_init) > threshold, 1, 0)
        lab = torch.sum(labs, dim=0)
        lab = torch.where(lab>0, 1, 0)
        assert torch.sum(lab) > 0, "Label is empty, cannot compute shape match metric."

        mismatch_visual = torch.abs(lab - pred)
        mismatch_number = torch.sum(mismatch_visual) / torch.sum(lab)
        if mismatch_number > 1.0:
            print(f"Warning: mismatch number is greater than 1.0: {mismatch_number}, which should not happen. This indicates a really bad prediction.")
        return mismatch_visual, mismatch_number

    if label.ndim == 4:
        collected_visuals = []
        collected_numbers = []
        for lab, pred in zip(label, prediction):
            mismatch_visual, mismatch_number = inner(lab, pred)
            collected_visuals.append(mismatch_visual)
            collected_numbers.append(mismatch_number)
        mismatch_visual = collected_visuals
        mismatch_number = torch.stack(collected_numbers).mean()
    elif label.ndim == 3:
        mismatch_visual, mismatch_number = inner(label, prediction)
    
    return mismatch_visual, mismatch_number