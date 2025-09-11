import torch
from einops import rearrange

class IndexLinear(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_groups: int, bias: bool = True):
        super().__init__()
        self.weights = torch.nn.Embedding(num_groups, dim_in * dim_out)
        self.bias = torch.nn.Embedding(num_groups, dim_out) if bias else None
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, x: torch.Tensor, idx: int):
        idx = torch.as_tensor(idx, device = x.device, dtype = torch.long)
        W = self.weights(idx).view(self.dim_out, self.dim_in)
        b = None if self.bias is None else self.bias(idx)
        return torch.nn.functional.linear(x, W, b)

class GroupLinear(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_groups: int, bias: bool = True):
        super().__init__()
        self.linear = IndexLinear(dim_in, dim_out, num_groups, bias)

    def forward(self, x: torch.Tensor, group_by: torch.LongTensor):
        # pre-allocate output tensor
        out = x.new_empty(*x.shape[:-1], self.linear.dim_out)
        # flat views
        xf = rearrange(x, '... d -> (...) d')
        of = rearrange(out, '... d -> (...) d')
        gf = rearrange(group_by, '... -> (...)')
        # determine which groups are present
        groups = gf.unique(sorted = False)
        # apply linear to groups
        for i, s in enumerate(groups):
            idx = (gf == s).nonzero().squeeze(1) # find all elements of the group
            of[idx] = self.linear(xf[idx], i)
        return out

class SegmentLinear(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, indices: torch.LongTensor, bias: bool = True):
        super().__init__()
        self.segments, num_groups = self.get_segments(indices)
        self.linear = IndexLinear(dim_in, dim_out, num_groups, bias)
        
    @staticmethod
    def get_segments(indices: torch.Tensor):
        unique_vals = indices.unique()
        segments = []
        start = 0
        current_val = indices[0]
        for i, val in enumerate(indices):
            if val != current_val:
                segments.append((current_val, slice(start, i)))
                start = i
                current_val = val
        segments.append((current_val, slice(start, len(indices))))
        return segments, len(unique_vals)

    def forward(self, x: torch.Tensor):
        out = x.new_empty(*x.shape[:-1], self.linear.dim_out)
        for idx, sl in self.segments:
            out[:, sl] = self.linear(x[:, sl], idx)
        return out