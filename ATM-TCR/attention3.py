import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ResidualBlock(nn.Module):
    def __init__(self, in_features, dropout):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.block(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.scaling_factor = math.sqrt(embed_dim)
        
    def forward(self, q, k, v):
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scaling_factor
        attention_weights = torch.softmax(attention_weights, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

class Net(nn.Module):
    def __init__(self, embedding, args):
        super(Net, self).__init__()
        self.num_amino = len(embedding)
        self.embedding_dim = len(embedding[0])
        self.embedding = nn.Embedding(self.num_amino, self.embedding_dim, padding_idx=self.num_amino-1)
        
        if not (args.blosum is None or args.blosum.lower() == 'none'):
            self.embedding = self.embedding.from_pretrained(torch.FloatTensor(embedding), freeze=False)

        # Positional Encoding
        self.pos_encoder_tcr = PositionalEncoding(self.embedding_dim, args.max_len_tcr)
        self.pos_encoder_pep = PositionalEncoding(self.embedding_dim, args.max_len_pep)
        
        # Self-attention layers
        self.attn_tcr = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=args.heads)
        self.attn_pep = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=args.heads)
        
        # Cross-attention layers
        self.cross_attn_t2p = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=args.heads)
        self.cross_attn_p2t = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=args.heads)
        
        # Layer normalization
        self.norm_tcr = nn.LayerNorm(self.embedding_dim)
        self.norm_pep = nn.LayerNorm(self.embedding_dim)
        self.norm_cross_tcr = nn.LayerNorm(self.embedding_dim)
        self.norm_cross_pep = nn.LayerNorm(self.embedding_dim)
        
        # Dense layers
        self.size_hidden1_dense = 2 * args.lin_size
        self.size_hidden2_dense = 1 * args.lin_size
        self.net_pep_dim = args.max_len_pep * self.embedding_dim
        self.net_tcr_dim = args.max_len_tcr * self.embedding_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.net_pep_dim + self.net_tcr_dim, self.size_hidden1_dense),
            nn.LayerNorm(self.size_hidden1_dense),
            ResidualBlock(self.size_hidden1_dense, args.drop_rate),
            nn.GELU(),
            nn.Dropout(args.drop_rate * 2),
            nn.Linear(self.size_hidden1_dense, self.size_hidden2_dense),
            nn.LayerNorm(self.size_hidden2_dense),
            ResidualBlock(self.size_hidden2_dense, args.drop_rate),
            nn.GELU(),
            nn.Dropout(args.drop_rate),
            nn.Linear(self.size_hidden2_dense, 1),
            nn.Sigmoid()
        )

    def forward(self, pep, tcr):
        # Embedding
        pep = self.embedding(pep)
        tcr = self.embedding(tcr)

        # Positional encoding
        pep = self.pos_encoder_pep(pep)
        tcr = self.pos_encoder_tcr(tcr)
        
        # Prepare for attention
        pep = torch.transpose(pep, 0, 1)
        tcr = torch.transpose(tcr, 0, 1)
        
        # Self attention
        pep_self, _ = self.attn_pep(pep, pep, pep)
        tcr_self, _ = self.attn_tcr(tcr, tcr, tcr)
        
        # Add & Norm after self attention
        pep_self = self.norm_pep(pep + pep_self)
        tcr_self = self.norm_tcr(tcr + tcr_self)
        
        # Cross attention
        tcr_cross, _ = self.cross_attn_t2p(tcr_self, pep_self, pep_self)
        pep_cross, _ = self.cross_attn_p2t(pep_self, tcr_self, tcr_self)
        
        # Add & Norm after cross attention
        tcr_final = self.norm_cross_tcr(tcr_self + tcr_cross)
        pep_final = self.norm_cross_pep(pep_self + pep_cross)
        
        # Transpose back
        pep_final = torch.transpose(pep_final, 0, 1)
        tcr_final = torch.transpose(tcr_final, 0, 1)
        
        # Linear layers
        pep_final = pep_final.reshape(-1, 1, pep_final.size(-2) * pep_final.size(-1))
        tcr_final = tcr_final.reshape(-1, 1, tcr_final.size(-2) * tcr_final.size(-1))
        peptcr = torch.cat((pep_final, tcr_final), -1).squeeze(-2)
        
        # Final prediction
        output = self.net(peptcr)
        
        return output