"""Restore official deterministic actors exactly; export adapted weights separately."""

from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from torch import nn


class FrozenNormalizer(nn.Module):
    def __init__(self, mean, denominator):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(np.array(mean, copy=True)))
        self.register_buffer("denominator", torch.as_tensor(np.array(denominator, copy=True)))

    def forward(self, observation):
        return (observation - self.mean) / self.denominator


class RestoredActor(nn.Module):
    def __init__(self, normalizer, mlp):
        super().__init__()
        self.obs_normalizer = normalizer
        self.mlp = mlp

    def forward(self, observation):
        return self.mlp(self.obs_normalizer(observation))


class ExecutionEnvelope(nn.Module):
    """Adapted policies include the same bounded execution transform used in PPO."""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, observation):
        return self.actor(observation).clamp(-5, 5)


def actor_from_onnx(path: str | Path) -> RestoredActor:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    nodes = list(model.graph.node)
    if [node.op_type for node in nodes] != [
        "Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"
    ]:
        raise ValueError("Unsupported actor graph; refusing a guessed weight conversion")
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    mean, denominator = arrays[nodes[0].input[1]], arrays[nodes[1].input[1]]
    if mean.shape != (1, 61) or denominator.shape != (1, 61):
        raise ValueError("Expected the frozen 61-dimensional normalizer")
    if not np.isfinite(denominator).all() or (denominator <= 0).any():
        raise ValueError("Invalid normalizer denominator")
    layers = []
    expected_widths = [(61, 512), (512, 256), (256, 128), (128, 14)]
    for index, node in enumerate(node for node in nodes if node.op_type == "Gemm"):
        attributes = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        if attributes.get("transB", 0) != 1 or attributes.get("transA", 0) != 0:
            raise ValueError("Unsupported Gemm transpose")
        if attributes.get("alpha", 1) != 1 or attributes.get("beta", 1) != 1:
            raise ValueError("Unsupported Gemm scaling")
        weight, bias = arrays[node.input[1]], arrays[node.input[2]]
        input_width, output_width = expected_widths[index]
        if weight.shape != (output_width, input_width) or bias.shape != (output_width,):
            raise ValueError("Actor dimensions differ from the official architecture")
        layer = nn.Linear(input_width, output_width)
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(np.array(weight, copy=True)))
            layer.bias.copy_(torch.from_numpy(np.array(bias, copy=True)))
        layers.append(layer)
        if index < 3:
            layers.append(nn.ELU())
    return RestoredActor(FrozenNormalizer(mean, denominator), nn.Sequential(*layers)).eval()


def export_actor(actor: nn.Module, destination: str | Path):
    destination = Path(destination)
    if not destination.name.endswith("_PhysX.onnx"):
        raise ValueError("Adapted models must have a distinct *_PhysX.onnx filename")
    destination.parent.mkdir(parents=True, exist_ok=True)
    device = next(actor.parameters()).device
    training = actor.training
    actor.eval()
    try:
        torch.onnx.export(
            actor, torch.zeros(1, 61, device=device), str(destination),
            input_names=["observation"], output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=13, dynamo=False,
        )
    finally:
        actor.train(training)
    onnx.checker.check_model(onnx.load(str(destination)))
