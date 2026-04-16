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