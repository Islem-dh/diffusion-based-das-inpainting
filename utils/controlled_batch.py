import numpy as np
from torch.utils.data import Sampler

class ControlledEBatch(Sampler):
    def __init__(self, weak_indices, medium_indices, strong_indices, extreme_indices, num_batches = 256, seed = 42):
        
        self.weak_indices = np.asarray(weak_indices)
        self.medium_indices = np.asarray(medium_indices)
        self.strong_indices = np.asarray(strong_indices)
        self.extreme_indices = np.asarray(extreme_indices)
        self.num_batches = num_batches
        self.seed = seed
        self.epoch = 0


    def set_epoch(self, epoch):
        self.epoch = epoch