import time
import argparse
import tiktoken
import torch
from torch.cpu import is_available
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class GPTDatasetV1(Dataset):

    def __init__(self, text, tokenizer, max_length, stride):

        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i+max_length]))
            self.target_ids.append(torch.tensor(token_ids[i+1:i+1+max_length]))
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader



class MultiHeadLatentAttention(nn.Module):

    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False, latent_dim=None):
        super().__init__()
        assert(d_out % num_heads) == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.latent_dim = latent_dim if latent_dim is not None else max(16, d_out//8)

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_DKV = nn.Linear(d_in, self.latent_dim, bias=qkv_bias) #down to latent c
        self.W_UK = nn.Linear(self.latent_dim, d_out, bias=qkv_bias) #up projection to k
        self.W_UV = nn.Linear(self.latent_dim, d_out, bias=qkv_bias) #up projection to v
        self.out_proj = nn.Linear(d_out, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # kv cache
        self.register_buffer("cache_c_kv", None, persistent=False)
        self.ptr_cur = 0

    def reset_cache(self):
        self.cache_c_kv = None
        self.ptr_cur = 0

    @staticmethod
    def _reshape_to_heads(x, head_dim, num_heads):
        """
            (b, num_tokens, d_out) -> (b, num_heads, num_tokens, head_dim)
        """
        b, num_tokens, _ = x.shape

        return x.view(b, num_tokens, num_heads, head_dim).transpose(1,2).contiguous()


    def forward(self, x, use_cache=False):
        b, num_tokens, _ = x.shape
        num_heads = self.num_heads
        head_dim = self.head_dim

        queries_all = self.W_query(x) # (b, tokens, d_out)
        latent_new = self.W_DKV(x) # (b, Tokens, latent_dim)


        if use_cache:
            if self.cache_c_kv is None:
                latent_total = latent_new
            else:
                latent_total = torch.cat([self.cache_c_kv, latent_new], dim=1) 
            self.cache_c_kv = latent_total
        else:
            latent_total = latent_new

        # up-projection
        keys_all = self.W_UK(latent_total) #(b, tokens, d_out)
        values_all = self.W_UV(latent_total)

        # reshaping
        queries_all = self._reshape_to_heads(queries_all, head_dim, num_heads)
        keys_all = self._reshape_to_heads(keys_all, head_dim, num_heads)
        values_all = self._reshape_to_heads(values_all, head_dim, num_heads) #(b, num_heads, tokens, head_dim)


        attention_scores = queries_all @ keys_all.transpose(2,3) 

        if use_cache:
            pos_idx = torch.arange(self.ptr_cur, self.ptr_cur + num_tokens, device=x.device).unsqueeze(-1)
            self.ptr_cur += num_tokens
        else:
            pos_idx = torch.arange(0, num_tokens, device=x.device).unsqueeze(-1)
            self.ptr_cur = 0

        col_idx = torch.arange(0, attention_scores.shape[-1], device=x.device).unsqueeze(0)
        causal_mask = pos_idx < col_idx

        attention_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -torch.inf)
        
        attention_weights = torch.softmax(attention_scores/keys_all.shape[-1]**0.5, dim=-1)
        assert keys_all.shape[-1] == self.head_dim
        attention_dropout = self.dropout(attention_weights)

        context_vectors = (attention_dropout @ values_all).transpose(1,2)

        context_vectors = context_vectors.contiguous().view(b, num_tokens, self.d_out)
        context_vectors = self.out_proj(context_vectors)

        return context_vectors



class LayerNorm(nn.Module):

    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean)/ torch.sqrt(variance + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), 
            GELU(), 
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadLatentAttention(d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], dropout=cfg["drop_rate"], num_heads=cfg["n_heads"], qkv_bias=cfg["qkv_bias"], latent_dim=cfg["latent_dim"])

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(emb_dim=cfg["emb_dim"])
        self.norm2 = LayerNorm(emb_dim=cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.ptr_current_pos = 0 
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x, use_cache=False):
        batch_size, seq_len = x.shape
        tok_embeds = self.tok_emb(x)

        if use_cache:
            pos_ids = torch.arange(self.ptr_current_pos, self.ptr_current_pos + seq_len, device=x.device, dtype=torch.long)
            self.ptr_current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=x.device, dtype=torch.long)
            self.ptr_current_pos = 0

        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        for blk in self.trf_blocks:
            x = blk(x, use_cache)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.ptr_current_pos = 0


def generate_text_simple_cached(model, idx, max_new_tokens, context_size=None, use_cache=True):
    model.eval()

    ctx_length = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            input_tokens = idx[:, -ctx_length:]
            input_tokens_length = input_tokens.size(1)

            model.reset_kv_cache()
            logits = model(idx, use_cache)

            for _ in range(max_new_tokens):
                nxt_tkn = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, nxt_tkn], dim=-1)
                logits = model(nxt_tkn, use_cache)

        else:
            for _ in range(max_new_tokens):
                idx = idx[:, -ctx_length:]
                logits = model(idx)
                nxt_tkn = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, nxt_tkn], dim=-1)

        return idx

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Run GPT with multi head latent attention.")
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--latent_dim", type=int, default=None, help="Latent dim for MLA")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),
        "emb_dim": args.emb_dim,    # Embedding dimension
        "n_heads": args.n_heads,    # Number of attention heads
        "n_layers": args.n_layers,  # Number of layers
        "drop_rate": 0.0,           # Dropout rate
        "qkv_bias": False,          # Query-Key-Value bias
        "latent_dim": args.latent_dim
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    model.eval()

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    token_ids = generate_text_simple_cached(
        model,
        encoded_tensor,
        args.max_new_tokens
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")

    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()