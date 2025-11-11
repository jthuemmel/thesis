import torch
from utils.components import *

class InterfaceNetwork(torch.nn.Module):
    def __init__(self, model, world):
        super().__init__()

        self.interface_network = torch.nn.ModuleList([
            InterfaceBlock(model.dim, 
                           dim_heads=model.dim_heads, 
                           num_blocks= model.num_compute_blocks, 
                           dim_ctx = model.dim_noise,
                           use_checkpoint=model.use_checkpoint)
            for _ in range(model.num_layers)
            ])

    def forward(self, 
                x: torch.Tensor,
                z: torch.Tensor, 
                ctx: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.interface_network(x = x, z = z, ctx = ctx)
    
class Perceiver(torch.nn.Module):
    def __init__(self, model, world):
        super().__init__()

        self.perceiver = InterfaceBlock(
            dim=model.dim, 
            num_blocks=model.num_layers,
            dim_ctx=model.dim_ctx, 
            dim_heads=model.dim_heads, 
            write_has_skip=False, 
            use_checkpoint=model.use_checkpoint
        )

        self.queries = torch.nn.Embedding(world.num_tokens, model.dim)

    def forward(self, 
                x: torch.Tensor,
                z: torch.Tensor, 
                ctx: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.queries.weight.expand(x.size(0), -1, -1)
        return self.perceiver(x = x, z = z, query = q, ctx = ctx)

class ViT(torch.nn.Module):
    def __init__(self, model, world):
        super().__init__()

        self.transformer = torch.nn.Sequential(*[
            TransformerBlock(
                dim =model.dim, 
                dim_heads=model.dim_heads, 
                dim_ctx = model.dim_noise
                ) for _ in range(model.num_layers)
            ])

    def forward(self, 
                x: torch.Tensor,
                z: torch.Tensor, 
                ctx: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, L = x.size(1), z.size(1)
        x = torch.cat([x, z], dim = 1)
        x = self.transformer(x, ctx = ctx)
        x, z = x.split([N, L], dim = 1)
        return x, z
    
class Flamingo(torch.nn.Module):
    def __init__(self, model, world):
        super().__init__()
        self.queries = torch.nn.Embedding(world.num_tokens, model.dim)
        self.compute = torch.nn.ModuleList([
            TransformerBlock(
                dim =model.dim, 
                dim_heads=model.dim_heads, 
                dim_ctx = model.dim_noise
                ) for _ in range(model.num_layers)
            ])
        self.write = TransformerBlock(
                dim =model.dim, 
                dim_heads=model.dim_heads, 
                dim_ctx = model.dim_noise
                )
        
    def forward(self, 
                x: torch.Tensor,
                z: torch.Tensor, 
                ctx: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.compute:
            z = block(q = z, kv = torch.cat([x, z], dim = 1), ctx = ctx)
        q = self.queries.weight.expand(x.size(0), -1, -1)
        q = self.write(q = q, kv = z)
        return q, z
