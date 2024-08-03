import pandas as pd
from torch import tensor
import torch
from torch.utils.data import Dataset

class Iris(Dataset):

    def __init__(self, file_name: str, train = True, device = "cuda"):
        super().__init__()

        self._device = device
        self._file_name: str = file_name
        self._train = train
        self._data = pd.read_csv(file_name).sample(frac=1)
        self._flower_name_map: dict[str, int] = {name: index for index, name in enumerate(list(set(self._data["species"])))}


    def __getitem__(self, index):
        index = index if self._train else index + 140
        sepal_length: float = list(self._data["sepal_length"])[index]
        sepal_width: float = list(self._data["sepal_width"])[index]
        petal_length: float = list(self._data["petal_length"])[index]
        petal_width: float = list(self._data["petal_width"])[index]
        species = self._flower_name_map[str(list(self._data["species"])[index])]

        return tensor([sepal_length, sepal_width, petal_length, petal_width], device=self._device, dtype = torch.float32), tensor([1 if i == species else 0 for i in range(3)], device=self._device, dtype=torch.float32)


    def __len__(self) -> int:
        return len(self._data["sepal_length"]) - 10 if self._train else 10


    def get_species(self, index) -> str|None:
        for name, i in self._flower_name_map.items():
            if i == index:
                return name
        
        return None
