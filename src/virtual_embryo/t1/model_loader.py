"""Load only the networks required by the selected prediction route."""
import json
from pathlib import Path
import torch
from ..paths import task_assets
from .model import TransitionModel


class PopulationPredictor:
    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir) if model_dir is not None else task_assets('t1')
        self.manifest = json.loads((self.model_dir/'manifest.json').read_text(encoding='utf-8'))
        self.config = self.manifest['config']
        self.n_genes = self.config['n_genes']
        self.models = {}

    def model(self, name):
        if name not in self.models:
            state = torch.load(self.model_dir/'inference'/f'{name}.pt', map_location='cpu', weights_only=True)
            net = TransitionModel(**self.config)
            net.load_state_dict(state['model'], strict=True)
            self.models[name] = net.eval().requires_grad_(False)
        return self.models[name]


def load_model(model_dir=None):
    return PopulationPredictor(model_dir)
