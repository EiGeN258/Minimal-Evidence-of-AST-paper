import torch

from learned_bias.model import BiasResNet
from learned_bias.streaming import StreamingBiasCorrector


def test_shape_and_constant_window_bias():
    model = BiasResNet()
    x = torch.randn(2, 200, 6)
    y = model(x)
    assert y.shape == x.shape
    assert torch.allclose(y[:, :1], y[:, -1:])
    assert 250000 <= model.parameter_count() <= 400000


def test_streaming_shape():
    model = BiasResNet()
    corrector = StreamingBiasCorrector(model, window=4)
    for _ in range(4):
        value, bias = corrector.update(torch.zeros(6).numpy())
    assert value.shape == (6,)
    assert bias.shape == (6,)
