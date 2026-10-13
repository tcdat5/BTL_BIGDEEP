"""ONNX nhỏ cho test phía cụm/dashboard (không cần torch): score = mean(input^2, axis=1)."""
from __future__ import annotations


def make_mean_square_onnx(path, d: int) -> None:
    import onnx
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", d])
    y = helper.make_tensor_value_info("score", TensorProto.FLOAT, ["batch"])
    nodes = [helper.make_node("Mul", ["input", "input"], ["sq"]),
             helper.make_node("ReduceMean", ["sq"], ["score"], axes=[1], keepdims=0)]
    model = helper.make_model(helper.make_graph(nodes, "mean_square", [x], [y]),
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
