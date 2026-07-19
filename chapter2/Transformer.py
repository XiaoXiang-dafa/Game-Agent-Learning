import torch
import torch.nn as nn
import math


# --- 基础模块 ---

class PositionalEncoding(nn.Module):
    """
    为输入序列的词嵌入向量添加位置编码。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000 ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够长的位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # pe (position encoding) 的大小为（max_len, d_model）
        pe = torch.zeros(max_len, d_model)

        # 偶数维度使用 sin, 奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将 pe 注册为buffer, 这样它就不会被视为模型参数, 但会随模型移动（例如to(device)）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.size(1) 是当前输入的序列长度
        # 将位置编码加入到输入向量上
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
   

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义 Q, K, V 和输出的线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 1. 计算注意力得分 （QK^T）
        attn_scores = torch.matmul(Q, K.transpose(-2, -1) / math.sqrt(self.d_k))

        # 2. 应用掩码 （如果提供）
        if mask is not None:

            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 3. 计算注意力权重（Softmax）
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 4. 加权求和（权重 * V）
        output = torch.matmul(attn_probs, V)
        return output
    
    def split_heads(self, x):
        # 将输入 x 的形状从 (batch_size, seq_length, d_model)
        # 变换为 （batch_size, num_heads, seq_length, d_k）
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # 将输入 x 的形状从（batch_size, num_heads, seq_length, d_k）
        # 变换为（batch_size, seq_length, d_model）
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        # 1. 对Q, K, V 进行线性变换
        Q = self.split_heads(self.w_q(Q))
        K = self.split_heads(self.w_k(K))
        V = self.split_heads(self.w_v(V))

        # 2.计算缩放点积注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 3. 合并多头输出并进行最终的线性变换
        output = self.w_o(self.combine_heads(attn_output))
        return  output
    

class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络模块
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
       super(PositionwiseFeedForward, self).__init__()
       self.linear1 = nn.Linear(d_model, d_ff)
       self.dropout = nn.Dropout(dropout)
       self.linear2 = nn.Linear(d_ff, d_model)
       self.relu = nn.ReLU()

    def forward(self, x):
        # x 形状: (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # 最终输出形状: (batch_size, seq_len, d_model)
        return x



# --- 编码器核心层 ---

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


# --- 解码器核心层 ---

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1. 掩码多头自注意力（对自己）
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 交叉注意力（对编码器输出）

        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 3. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


# ============================================================
# Mask 工具函数
# ============================================================

def generate_padding_mask(seq, pad_idx=0):
    """
    生成 padding mask，屏蔽 <pad> token 的位置。

    参数:
        seq: (batch_size, seq_len) 输入序列，每行是 token id
        pad_idx: padding token 的索引，默认为 0

    返回:
        mask: (batch_size, 1, 1, seq_len)，pad 位置为 0，其余为 1
              可直接传入 MultiHeadAttention 的 mask 参数
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def generate_subsequent_mask(sz):
    """
    生成 look-ahead mask（下三角矩阵），用于解码器的自回归注意力。
    确保位置 i 只能看到位置 j <= i 的信息。

    参数:
        sz: 序列长度

    返回:
        mask: (1, 1, sz, sz)，上三角为 0，下三角（含对角线）为 1
    """
    return torch.tril(torch.ones(sz, sz)).bool().unsqueeze(0).unsqueeze(0)


# ============================================================
# 编码器
# ============================================================

class Encoder(nn.Module):
    """
    完整的 Transformer 编码器，由 num_layers 个 EncoderLayer 堆叠而成。
    """
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        """
        参数:
            x: (batch_size, src_seq_len, d_model) 词嵌入 + 位置编码后的输入
            mask: (batch_size, 1, 1, src_seq_len) padding mask
        返回:
            (batch_size, src_seq_len, d_model) 编码后的表示
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ============================================================
# 解码器
# ============================================================

class Decoder(nn.Module):
    """
    完整的 Transformer 解码器，由 num_layers 个 DecoderLayer 堆叠而成。
    """
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        参数:
            x: (batch_size, tgt_seq_len, d_model) 目标序列的嵌入 + 位置编码
            encoder_output: (batch_size, src_seq_len, d_model) 编码器输出
            src_mask: padding mask for cross-attention
            tgt_mask: look-ahead + padding mask for self-attention
        返回:
            (batch_size, tgt_seq_len, d_model) 解码后的表示
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


# ============================================================
# 完整 Transformer 模型
# ============================================================

class Transformer(nn.Module):
    """
    完整的 Transformer 序列到序列模型。

    使用示例:
        model = Transformer(src_vocab_size=10000, tgt_vocab_size=10000)
        src = torch.randint(0, 10000, (2, 50))   # (batch=2, seq_len=50)
        tgt = torch.randint(0, 10000, (2, 40))   # (batch=2, seq_len=40)

        src_mask = model.generate_padding_mask(src)   # 屏蔽 src 中的 pad
        tgt_mask = model.generate_tgt_mask(tgt)       # 屏蔽 tgt 中的 pad + 未来信息

        output = model(src, tgt, src_mask, tgt_mask)  # (2, 40, 10000)
    """
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 pad_idx: int = 0):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        # 位置编码（encoder 和 decoder 共用）
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

        # Encoder / Decoder
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)

        # 输出投影层：d_model → tgt_vocab_size
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # 参数初始化
        self._init_parameters()

    def _init_parameters(self):
        """Xavier/Glorot 初始化，帮助训练稳定"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_padding_mask(self, seq):
        """生成 padding mask，屏蔽 <pad> token"""
        return (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def generate_tgt_mask(self, tgt):
        """
        生成解码器的完整目标掩码：
        - padding mask：屏蔽 <pad> token
        - subsequent mask：屏蔽未来位置（look-ahead）
        两者取交集（同时满足才可见）
        """
        tgt_seq_len = tgt.size(1)
        pad_mask = self.generate_padding_mask(tgt)                    # (B, 1, 1, T)
        sub_mask = generate_subsequent_mask(tgt_seq_len).to(tgt.device)  # (1, 1, T, T)
        return pad_mask & sub_mask                                    # (B, 1, T, T)

    def encode(self, src, src_mask=None):
        """编码源序列（训练和推理共用）"""
        # 嵌入 + 缩放 + 位置编码
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        return self.encoder(x, src_mask)

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """解码目标序列（训练和推理共用）"""
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        return self.decoder(x, encoder_output, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        参数:
            src: (batch_size, src_seq_len) 源语言 token id
            tgt: (batch_size, tgt_seq_len) 目标语言 token id
            src_mask: padding mask for encoder
            tgt_mask: combined mask for decoder self-attention
        返回:
            (batch_size, tgt_seq_len, tgt_vocab_size) logits
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_projection(decoder_output)