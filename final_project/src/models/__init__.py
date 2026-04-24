from .mlp import MLPAE
from .cnn1d import CNN1DAE
from .rnn import RNNAE

MODELS = {"mlp": MLPAE, "cnn1d": CNN1DAE, "rnn": RNNAE}


def build_model(name: str, latent_dim: int = 16, window_len: int = 128):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")
    return MODELS[name](window_len=window_len, latent_dim=latent_dim)
