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

class Net(nn.Module):
    def __init__(self, embedding, args):
        super(Net, self).__init__()
        self.num_amino = len(embedding)
        self.embedding_dim = len(embedding[0])
        self.embedding = nn.Embedding(self.num_amino, self.embedding_dim, padding_idx=self.num_amino-1)
        
        self.positional_encoding_pep = PositionalEncoding(self.embedding_dim, max_len=args.max_len_pep)
        self.positional_encoding_tcr = PositionalEncoding(self.embedding_dim, max_len=args.max_len_tcr)
        
        if not (args.blosum is None or args.blosum.lower() == 'none'):
            self.embedding = self.embedding.from_pretrained(torch.FloatTensor(embedding), freeze=False)

        self.pos_encoder_tcr = PositionalEncoding(self.embedding_dim, args.max_len_tcr)
        self.pos_encoder_pep = PositionalEncoding(self.embedding_dim, args.max_len_pep)
        
        self.attn_tcr = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=args.heads)
        self.attn_pep = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=args.heads)
        
        self.size_hidden1_dense = 2 * args.lin_size
        self.size_hidden2_dense = 1 * args.lin_size
        self.net_pep_dim = args.max_len_pep * self.embedding_dim
        self.net_tcr_dim = args.max_len_tcr * self.embedding_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.net_pep_dim + self.net_tcr_dim,
                      self.size_hidden1_dense),
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
        pep = self.embedding(pep)  # batch * len * dim
        tcr = self.embedding(tcr)  # batch * len * dim

        pep = self.pos_encoder_pep(pep)
        tcr = self.pos_encoder_tcr(tcr)
        
        # Prepare for attention (transpose)
        pep = torch.transpose(pep, 0, 1)
        tcr = torch.transpose(tcr, 0, 1)
        
        # Attention
        pep, pep_attn = self.attn_pep(pep, pep, pep)
        tcr, tcr_attn = self.attn_tcr(tcr, tcr, tcr)
        
        # Transpose back
        pep = torch.transpose(pep, 0, 1)
        tcr = torch.transpose(tcr, 0, 1)
        
        # Linear
        pep = pep.reshape(-1, 1, pep.size(-2) * pep.size(-1))
        tcr = tcr.reshape(-1, 1, tcr.size(-2) * tcr.size(-1))
        peptcr = torch.cat((pep, tcr), -1).squeeze(-2)
        peptcr = self.net(peptcr)
        
        return peptcr