import torch
from torch import einsum, nn

def sdp_attention(q, k, v, mask=None):
    head_dim = q.size(-1)

    attn_scores = einsum('bhik, bhjk -> bhij', q, k) / (head_dim ** 0.5)

    if mask is not None:
        # print(f"{mask.shape=}")
        mask = mask.unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = torch.softmax(attn_scores, dim=-1)

    output = einsum('bhij, bhjd -> bhid', attn_weights, v)
    return attn_weights, output

class Attention(nn.Module):
    def __init__(self, dim:int, nheads:int):
        super().__init__()
        assert dim % nheads == 0, ValueError()

        self.head_dim = dim//nheads
        self.nhead = nheads
        self.dim = dim

        self.wq = nn.Parameter(
            torch.randn(dim, dim)
        )
        self.wk = nn.Parameter(
            torch.randn(dim, dim)
        )
        self.wv = nn.Parameter(
            torch.randn(dim, dim)
        )
        self.wo = nn.Parameter(
            torch.randn(dim, dim)
        )

    def forward(self, x:torch.Tensor, mask:torch.Tensor | None = None):
        batch_size, seq_len, _ = x.shape

        q = torch.matmul(x, self.wq)
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)

        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        attn_weight, attn_output = sdp_attention(q, k, v, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        output = torch.matmul(attn_output, self.wo)
        return output, (attn_weight, attn_output)

if "__main__" in __name__:
    att = Attention(12, 4)

    x = torch.randn(1, 10, 12)
    mask = (torch.arange(0, 10) < 5)[None, :]

    o, dummy = att(x, mask)
    print((dummy[0][0, 0] != 0).int())
