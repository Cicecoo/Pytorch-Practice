import torch
from torch.onnx import utils as jit_utils


opset_version = 20

class CustomErfc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # ctx.save_for_backward(x)
        return torch.erfc(x)
    @staticmethod
    def symbolic(g, x):
        return g.op("onnx::Erfc", x)
    
torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::erfc",
    symbolic_fn=CustomErfc.symbolic,
    opset_version=opset_version,
)