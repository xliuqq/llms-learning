import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        '''
        :param d_in: 嵌入向量的维度，即输入词元的向量维度
        :param d_out: 输出向量的维度
        :param qkv_bias: 是否使用偏置项
        '''
        super().__init__()

        # 使用 Linear 层，提供优化的权重初始化方案，且当偏置禁用时，更高效执行乘法
        self.query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        # 输入 x 的维度为 [B, D_in]，输出 x 的维度为 [B, D_out]
        q = self.query(x) # [B, D_out]   x @ query
        k = self.key(x) # [B, D_out]
        v = self.value(x) # [B, D_out]

        atten_scores = q @ k.T
        atten_weights = torch.softmax(atten_scores / (k.shape[-1] ** 0.5), dim=-1)

        all_context_vecs = atten_weights @ v  # 计算上下文向量
        return all_context_vecs


class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        
        # 模型训练时不会更新（即调用 optimizer.step() 后该组参数不会变化，只可人为地改变它们的值），但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # 将维度1和维度2转置，将批维度保持在第0维度
        # 对每个批次下的矩阵做乘法
        atten_scores = queries @ keys.transpose(1, 2) 
        # 以下划线结尾的操作，直接应用于原数组
        atten_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # keys.shape[-1] == d_out
        atten_weights = torch.softmax(atten_scores / (keys.shape[-1] ** 0.5), dim=-1)
        atten_weights = self.dropout(atten_weights)

        context_vecs = atten_weights @ values
        return context_vecs
    

class MultiHeadAttentionWrapper(nn.Module):
    """
    Docstring for MultiHeadAttentionWrapper

    使用多个 CausalSelfAttention 头来增强模型的表达能力。输出的上下文向量维度为 d_out * num_heads。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.headers = nn.ModuleList([
            CausalSelfAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        # 对每个头进行注意力计算，并将结果拼接在一起
        head_outputs = [head(x) for head in self.headers]
        concatenated = torch.cat(head_outputs, dim=-1)
        return concatenated
    

class MultiHeadAttention(nn.Module):
    """
    Docstring for MultiHeadAttention

    使用单个矩阵计算多个注意力头。 输出的上下文向量维度为 d_out 
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads) == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads      # 减少投影维度以匹配所需要的输出维度
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 最终的线性投影层，组合头的输出
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # 张量形状 (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 通过添加一个 num_heads 维度隐式分割矩阵，展开最后一个维度。(b, num_tokens, d_out) ->  (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        #  (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        atten_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        atten_scores.masked_fill_(mask_bool, -torch.inf)

        atten_weights = torch.softmax(atten_scores / (keys.shape[-1] ** 0.5), dim=-1)
        atten_weights = self.dropout(atten_weights)

        # (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        context_vecs = (atten_weights @ values).transpose(1, 2)

        # 将最后两个维度展平，得到最终的上下文向量 (b, num_tokens, d_out)
        context_vecs = context_vecs.contiguous().view(b, num_tokens, self.d_out)

        # 可选的线性投影
        context_vecs = self.out_proj(context_vecs)  # context_vecs @ out_proj
        return context_vecs
