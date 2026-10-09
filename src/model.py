import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

class MyModel(nn.Module):
    _file_name: str

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        self._file_name = time.strftime("%Y-%m-%d_%I-%M.pt")

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

    def save(self):
        MODELS_DIR.mkdir(exist_ok=True)
        torch.save(self.state_dict(), MODELS_DIR / self._file_name)

    def load(self, file_path: str):
        self.load_state_dict(torch.load(file_path))
