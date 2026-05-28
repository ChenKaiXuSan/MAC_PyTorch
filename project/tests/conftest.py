import torch
import pytest


@pytest.fixture(scope='session')
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def rng():
    g = torch.Generator()
    g.manual_seed(42)
    return g
