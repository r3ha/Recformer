from typing import List
from torch.utils.data import Dataset
from collator import PretrainDataCollatorWithPadding


class ClickDataset(Dataset):
    def __init__(self, dataset: List, collator: PretrainDataCollatorWithPadding):
        super().__init__()

        self.dataset = dataset
        self.collator = collator

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        
        return self.dataset[index]

    def collate_fn(self, data):
        # A list of a dict form of user interaction history [{'items': [item_1, item_2, ..., item_m]}, ... , {...}]
        return self.collator([{'items': line} for line in data])
