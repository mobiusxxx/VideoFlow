import torch
from torch import nn, einsum
from einops import rearrange


class RelPosEmb(nn.Module):
    def __init__(
            self,
            max_pos_size,
            dim_head
    ):
        super().__init__()
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, dim_head)
        self.rel_width = nn.Embedding(2 * max_pos_size - 1, dim_head)

        deltas = torch.arange(max_pos_size).view(1, -1) - torch.arange(max_pos_size).view(-1, 1)
        rel_ind = deltas + max_pos_size - 1
        self.register_buffer('rel_ind', rel_ind)

    def forward(self, q):
        batch, heads, h, w, c = q.shape
        height_emb = self.rel_height(self.rel_ind[:h, :h].reshape(-1))
        width_emb = self.rel_width(self.rel_ind[:w, :w].reshape(-1))

        height_emb = rearrange(height_emb, '(x u) d -> x u () d', x=h)
        width_emb = rearrange(width_emb, '(y v) d -> y () v d', y=w)

        height_score = einsum('b h x y d, x u v d -> b h x y u v', q, height_emb)
        width_score = einsum('b h x y d, y u v d -> b h x y u v', q, width_emb)

        return height_score + width_score


class Attention(nn.Module):
    def __init__(
        self,
        *,
        args,
        dim,
        max_pos_size = 100,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)

        self.pos_emb = RelPosEmb(max_pos_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k = self.to_qk(fmap).chunk(2, dim=1)  # chunk(2, dim=1) 将返回的张量在第一个维度上（通道维度）进行切片，分成两部分，分别赋给 q 和 k。

        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=heads), (q, k))  # rearrange 函数对 q 和 k 进行形状变换。
        q = self.scale * q  # 为了调整注意力的强度。

        if self.args.position_only:
            sim = self.pos_emb(q)  # 如果 position_only 为 True，则使用位置嵌入 self.pos_emb 对查询张量 q 进行处理，得到注意力分数张量 sim。

        elif self.args.position_and_content:
            """同时考虑位置和内容信息。"""
            sim_content = einsum('b h x y d, b h u v d -> b h x y u v', q, k)
            sim_pos = self.pos_emb(q)
            sim = sim_content + sim_pos

        # else:
        sim = einsum('b h x y d, b h u v d -> b h x y u v', q, k)  # einsum 是爱因斯坦求和的缩写，用于执行张量的乘法和求和。计算了查询张量 q 和键张量 k 的乘积，得到注意力分数张量 sim。

        sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')  # 重新排列张量维度的函数，通常用于执行张量的转置、变形等操作。
        attn = sim.softmax(dim=-1)

        return attn


class Aggregate(nn.Module):
    def __init__(
        self,
        args,
        dim,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        return out


if __name__ == "__main__":
    att = Attention(dim=128, heads=1)
    fmap = torch.randn(2, 128, 40, 90)
    out = att(fmap)

    print(out.shape)
