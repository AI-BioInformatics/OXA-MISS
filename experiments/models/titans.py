import torch
import torch.nn as nn
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
    MemoryAttention,
    MIL_MAC
)

class TITANS(nn.Module):  
    def __init__(
        self,
        input_dim=1024, 
        inner_dim=64,
        dropout=0.0,
        output_dim=4,
        **kwargs
    ):
        super().__init__() 
        # Mapping custom parameter names to the expected ones in MemoryAsContextTransformer
        mapped_params = {
            "dim": inner_dim,  # input_dim -> dim
            "depth": kwargs.get("num_layers", 2),  # num_layers -> depth
            "num_tokens": kwargs.get("num_tokens", 256),  # num_tokens
            "segment_len": kwargs.get("segment_len", 512),  # segment_len  #rimettere a 32
            "num_persist_mem_tokens": kwargs.get("num_persist_mem_tokens", 16),  # num_persist_mem_tokens
            "num_longterm_mem_tokens": kwargs.get("num_longterm_mem_tokens", 16),  # num_longterm_mem_tokens
            "neural_memory_layers": kwargs.get("neural_memory_layers", (2,)),  # neural_memory_layers
            "neural_memory_segment_len": kwargs.get("neural_memory_segment_len", 16),  # neural_memory_segment_len
            "neural_memory_batch_size": kwargs.get("neural_memory_batch_size",512), #128 as default  # neural_memory_batch_size
            "neural_mem_gate_attn_output": kwargs.get("neural_mem_gate_attn_output", False),  # neural_mem_gate_attn_output
            "neural_mem_weight_residual": kwargs.get("neural_mem_weight_residual", True),  # neural_mem_weight_residual
            "use_flex_attn": kwargs.get("use_flex_attn", False),  # use_flex_attn
            "sliding_window_attn": kwargs.get("sliding_window_attn", False),  # sliding_window_attn
            "output_classes": output_dim,  # output_dim
            "num_residual_streams": kwargs.get("num_residual_streams", 1),  # num_residual_streams 4
            "neural_memory_kwargs": kwargs.get("default_step_transform_max_lr", {"default_step_transform_max_lr": 0.1}),  # default_step_transform_max_lr
            # "neural_memory_kwargs": kwargs.get("max_grad_norm", {"max_grad_norm":0.1}),  # max_grad_norm
            # "neural_memory_kwargs": kwargs.get("momentum", {"momentum":False}),  # momentum
        }
        self.inner_proj=nn.Linear(input_dim, inner_dim)
        self.mil_mac= MIL_MAC(**mapped_params)
        # print("rimettere segment_len a 32")
    def forward(self, data):
        tmp_output = []
        x = data['patch_features']  # x is a dictionary with key 'patch_features'
        mask = data['mask']
        x = x[~mask.bool()].unsqueeze(0)
        x = self.inner_proj(x)
        x=self.mil_mac(x) #[:,:64,:] [:,:1024,:]
        output = {'output': x}
        return output