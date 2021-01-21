import torch
import torch.nn as nn

from gander.models.gan import _gradient_penalty, _random_sample_line_segment


class MockDescriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([5.0]))

    def forward(self, x, _, __):
        return self.theta * x * x


def test_gradient_penalty_loss():
    """
    A test to verify that the loss based on second derivatives works correctly
    """
    torch.autograd.set_detect_anomaly(True)
    descriminator = MockDescriminator()

    params = list(descriminator.parameters())
    assert len(params) == 1
    assert params[0] == 5

    x_hat = torch.tensor([[3.0, -2.0], [5.0, 7.0]])
    gp, _ = _gradient_penalty(x_hat, descriminator, None, None)

    # ∂f/∂x = θ * 2 * x = 5.0 * 2 * [[3.0, -2.0], [5.0, 7.0]] = [[30.0, -20.0], [50.0, 70.0]]
    # |∂f/∂x| = [[(30.0**2 + -20.0**2)^0.5], [(50.0**2 + 70.0**2)^0.5]] = [[36.05551], [86.0233]]
    # (|∂f/∂x| - 1)^2 = [[1228.88897], [7228.95304]]
    print("GP", gp)
    assert torch.allclose(gp, torch.tensor([1228.88897, 7228.9536]), rtol=1e-3)

    x = torch.tensor([[7.0, 11.0], [3.0, 5.0]])
    f = descriminator(x, None, None)

    # Verify that the second derivatives are correctly computed and summed.
    v1 = 51.0
    v2 = 1715.5842
    l = f.mean()
    l.backward(retain_graph=True)
    assert torch.allclose(params[0].grad, torch.tensor([v1]))
    descriminator.zero_grad()

    l = gp.mean()
    l.backward(retain_graph=True)
    assert torch.allclose(params[0].grad, torch.tensor([v2]))
    descriminator.zero_grad()

    l = f.mean() + gp.mean()
    l.backward(retain_graph=True)
    assert torch.allclose(params[0].grad, torch.tensor([v1 + v2]))
    descriminator.zero_grad()


def test_random_sample_line_segment():
    x1 = torch.tensor(
        [
            [
                [[1, 2, 3, 9], [3, 4, 5, 6]],
                [[1, 2, 3, 9], [3, 4, 5, 6]],
                [[1, 2, 3, 9], [3, 4, 5, 6]],
            ]
        ]
    )
    assert x1.shape == (1, 3, 2, 4)
    x2 = torch.tensor(
        [
            [
                [[2, 3, 4, 10], [4, 5, 6, 7]],
                [[2, 3, 4, 10], [4, 5, 6, 7]],
                [[2, 3, 4, 10], [4, 5, 6, 7]],
            ]
        ]
    )

    y = _random_sample_line_segment(x1, x1)
    assert y.shape == x1.shape
    assert torch.allclose(y, x1)

    y = _random_sample_line_segment(x1, x2)
    assert y.shape == x1.shape
    assert x1[0, 0, 0, 0] <= y[0, 0, 0, 0] and y[0, 0, 0, 0] <= x2[0, 0, 0, 0]
