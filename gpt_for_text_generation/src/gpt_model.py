import torch
import torch.nn as nn
from Transformer_block import *

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        batch_size, seq_len = x.shape
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(torch.arange(seq_len))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
# batch = torch.tensor([[6109,  3626,  6100,   345],     
#  [6109,  1110,  6622,   257]])
# # out = model(batch)
# # print(out.shape)
# val = sum([p.numel() for p in model.parameters()])
# # print(val)
# # print(model.tok_emb.weight.shape)
# # print(model.out_head.weight.shape)
# # total_val = val - sum([p.numel() for p in model.out_head.parameters()])
# # print(total_val)
# total_size = val * 4
# total_size_mb = total_size / (1024)**2
# print(total_size_mb)