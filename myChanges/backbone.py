import torch.nn as nn
import torch

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        return self.fn(self.norm(x))

class FeedForward_vitsegnet(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        qkv_list = []
        for t in qkv:
            # Preparing each dimension's value
            b, n, hd = t.shape
            h = self.heads
            d = hd // h

            # First, reshape the tensor
            t = t.view(b, n, h, d)
            # Then, permute the dimensions
            t = t.permute(0, 2, 1, 3)
            qkv_list.append(t)
        
        q, k, v = qkv_list
        k_transposed = k.permute(0, 1, 3, 2)
        dots = torch.matmul(q, k_transposed) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)

        # Preparing each dimension's value
        b, h, n, d = out.shape
        hd = h * d

        # First, reshape the tensor
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(b, n, hd)
        out = self.to_out(out)

        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.depth = depth

        self.layerAttention = nn.ModuleList([])
        self.layerFeedForward = nn.ModuleList([])

        for _ in range(depth):

            attention_layer = PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))
            feedforward_layer = PreNorm(dim, FeedForward_vitsegnet(dim, mlp_dim, dropout = dropout))

            self.layerAttention.append(
                attention_layer,
            )
            self.layerFeedForward.append(
                feedforward_layer,
            )

    def forward(self, x):

        for i in range(self.depth):
            for iter, attention in enumerate(self.layerAttention):
                if iter == i:
                    x = attention(x) + x

            for iter, feedforward in enumerate(self.layerFeedForward):
                if iter == i:
                    x = feedforward(x) + x

        return x

class VitSegNet(nn.Module):
    def __init__(self,
                image_size=144,
                patch_size=8,
                channels=64,
                dim=512,
                depth=5,
                heads=16,
                output_channels=1024,
                expansion_factor=4,
                dim_head=64,
                dropout=0.,
                emb_dropout=0.,
                is_with_shared_mlp=True):  # mlp_dim is corresponding to expansion factor
        super().__init__()
        image_height, image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)

        assert image_height % self.patch_height == 0 and image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // self.patch_height) * (image_width // self.patch_width)
        patch_dim = channels * self.patch_height * self.patch_width
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Linear(patch_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        mlp_dim = int(dim*expansion_factor)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.temp_h = int(image_size/patch_size)
        
        out_in_channels = int(dim/(patch_size**2))

        if is_with_shared_mlp:
            self.is_with_shared_mlp = True
            self.shared_mlp = nn.Conv2d(in_channels=out_in_channels, out_channels=output_channels, kernel_size=1)
        else:
            self.is_with_shared_mlp = False
            self.shared_mlp = nn.Identity()

    def forward(self, img):

        ''' 
        ERROR:

        /home/username/.local/lib/python3.10/site-packages/torch/onnx/utils.py:1957: FutureWarning: 'torch.onnx.symbolic_opset9._cast_Float' is deprecated in version 2.0 and will be removed in the future. Please Avoid using this function and create a Cast node instead.
        return symbolic_fn(graph_context, *inputs, **attrs)
        
        DON'T DO THIS:

                h = int(img.shape[2] / patch_height)
                w = int(img.shape[3] / patch_width)

        INSTEAD DO THIS:

                h = img.shape[2] // patch_height
                w = img.shape[3] // patch_width
        '''

        patch_height = self.patch_height
        patch_width = self.patch_width
        h = img.shape[2] // patch_height
        w = img.shape[3] // patch_width
        c = img.shape[1]
        intermediate_step = img.reshape(1, c, h, patch_height, w, patch_width)
        intermediate_step = intermediate_step.permute(0, 2, 4, 3, 5, 1)
        intermediate_step = intermediate_step.reshape(1, h*w, patch_height*patch_width*64)
        img = intermediate_step

        x = self.to_patch_embedding(img)
        _, n, _ = x.shape
        
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)

        h = self.temp_h
        p1 = patch_height
        p2 = patch_width
        b = x.shape[0]
        w = x.shape[1] // h
        c = x.shape[2] // (p1*p2)
        x = x.reshape(b, h, w, p1, p2, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(b, c, h*p1, w*p2)

        # Since we don't use MLP layer we should remove it to make our model graph a little simple
        if self.is_with_shared_mlp:
            x = self.shared_mlp(x)

        return x