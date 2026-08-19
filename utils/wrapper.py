from torch.utils.data import Dataset
class Source(Dataset):
    def __init__(self, dataset, source):
        self.dataset = dataset
        self.source = source

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.source    