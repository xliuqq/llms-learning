import torch
import torch.nn as nn

from SelfAttention import MultiHeadAttention

class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # 使用有偏方差估计，与 GPT-2 模型的归一化层兼容
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean)/torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
                    torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
               )
        )
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 线性层（输出维度*4） -> GELU 层 -> 线性层（输出维度/4）
        self.layers = nn.Sequential(
            
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4),
            GELU(),
            nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    """
    Docstring for TransformerBlock.

    x -> 层归一化1 -> 掩码多头注意力 -> dropout -(+shortcut_x) -> y
    
    y -> 层归一化2 -> FeedForward -> dropout -(+shortcut_y) -> output
    """
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        self.ff = FeedForward(cfg)
        self.norm2 = LayerNorm(cfg["emb_dim"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)  # TODO: 这里为什么用同一个 dropout 层
        x = x + shortcut

        return x
    

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 词元嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 位置嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # 最终层归一化
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 线性输出层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # seq_len：输入序列的长度，batch_size：输入批次的大小
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # print("tok_embeds shape:", tok_embeds.shape)  # (batch_size, seq_len, emb_dim)

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # print("pos_embeds shape:", pos_embeds.shape)  # (seq_len, emb_dim)

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)  

        # print("After transformer blocks, x shape:", x.shape)  # (batch_size, seq_len, emb_dim)
        x = self.final_norm(x)
        # print("After final norm, x shape:", x.shape)  # (batch_size, seq_len, emb_dim)

        # 非归一化概率
        logits = self.out_head(x)

        return logits

