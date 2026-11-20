import pytest
import torch

from src.visionforge.models.cnn import CNN


@pytest.mark.parametrize(
    "channels, expected_spatial",
    [((32, 64, 128), 4), ((8, 16), 8)],
)
def test_features_halve_spatial_size_per_block(channels, expected_spatial):
    model = CNN(channels=channels, hidden=16)

    features = model.features(torch.randn(2, 3, 32, 32))

    assert features.shape == (2, channels[-1], expected_spatial, expected_spatial)


def test_forward_returns_one_logit_per_class():
    model = CNN(num_classes=10, channels=(8, 16), hidden=16)

    assert model(torch.randn(4, 3, 32, 32)).shape == (4, 10)


def test_rejects_inputs_that_do_not_match_the_flattened_size():
    model = CNN(channels=(8, 16), hidden=16)

    with pytest.raises(RuntimeError):
        model(torch.randn(2, 3, 28, 28))


def test_eval_mode_is_deterministic():
    model = CNN(channels=(8, 16), hidden=16, dropout=0.5).eval()
    x = torch.randn(4, 3, 32, 32)

    with torch.inference_mode():
        assert torch.equal(model(x), model(x))


def test_backward_pass_produces_gradients_for_every_parameter():
    model = CNN(channels=(8, 16), hidden=16)
    loss = model(torch.randn(4, 3, 32, 32)).sum()

    loss.backward()

    assert all(p.grad is not None for p in model.parameters())
