import numpy as np
import torch

from agcnet.model import AGCNet
from agcnet.streaming import StreamingAGCNet


def test_model_shape():
    model = AGCNet()
    x = torch.randn(2, 512, 6)
    y = model(x)
    assert y.shape == x.shape


def test_streaming_shape():
    model = AGCNet()
    corrector = StreamingAGCNet(model, window=16, horizon=2)
    for _ in range(18):
        value, compensation = corrector.update(np.zeros(6, dtype=np.float32))
    assert value.shape == (6,)
    assert compensation.shape == (6,)
