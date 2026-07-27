import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.normalizer import normalize_shift

class DasDataset(Dataset):
    def __init__(self, paths):
        self.file_paths = list(paths)
        self.files =[]
        self.patches = []
        self.index = []
        for file, file_path in enumerate(self.file_paths):
            f = h5py.File(file_path, "r")
            self.files.append(f)
            self.patches.append(f["patches"])
            num_patches = f["patches"].shape[0]
            #with h5py.File(file_path,"r") as f:
                #num_patches = f["patches"].shape[0]
            for patch in range(num_patches):
                self.index.append((file,patch))
        

    def __len__(self):
        return len(self.index)

    '''def get_file(self,file):
        if self.open_files[file] is None:
            self.open_files[file]=h5py.File(self.file_paths[file],"r")
        return self .open_files[file]'''

    def __getitem__(self,i):
        file,patch_index = self.index[i]
        #f = self.get_file(file)
        patch = np.asarray(self.patches[file][patch_index])
        patch= torch.tensor(patch).unsqueeze(0)
        patch, mu = normalize_shift(patch)
        return patch, mu
    
    def close(self):
        for f in self.files:
            f.close()      