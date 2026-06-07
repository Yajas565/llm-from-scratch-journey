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



class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, max_seq_len=None, window_size=None):
        super().__init__()
        assert(d_out % num_heads) == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # kv cache
        self.max_seq_len = max_seq_len or context_length
        self.window_size = window_size or self.max_seq_len
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)


    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape

        if use_cache:
            assert num_tokens <= self.window_size, (f"input chunk size {num_tokens} exceeds the window size {self.window_size}")

        queries_new = self.W_query(x)
        keys_new = self.W_key(x)
        values_new = self.W_value(x)

        queries_new = queries_new.view(b, num_tokens, self.num_heads, self.head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)

        queries_new = queries_new.transpose(1,2)
        keys_new = keys_new.transpose(1,2)
        values_new = values_new.transpose(1,2)

        if use_cache:
            if self.cache_k is None or self.cache_k.size(0) != b:
                self.cache_k = torch.zeros((b,self.num_heads,self.window_size,self.head_dim), device=x.device)
                self.cache_v = torch.zeros_like(self.cache_k)
                self.ptr_cur = 0
            
            if self.ptr_cur + num_tokens > self.window_size:
                overflow = self.ptr_cur + num_tokens - self.window_size
                self.cache_k[:, :, :-overflow, :] = self.cache_k[:, :, overflow:, :].clone()
                self.cache_v[:, :, :-overflow, :] = self.cache_v[:, :, overflow:, :].clone()

            self.cache_k[:, :, self.ptr_cur:self.ptr_cur+num_tokens, :] = keys_new
            self.cache_v[:, :, self.ptr_cur:self.ptr_cur+num_tokens, :] = values_new 
            self.ptr_cur += num_tokens

            keys = self.cache_k[:, :, :self.ptr_cur, :]
            values = self.cache_v[:, :, :self.ptr_cur, :]

        else:
            keys, values = keys_new, values_new
            self.ptr_cur = 0

        attention_scores = queries_new @ keys.transpose(2,3) 

        k = attention_scores.size(-1)
        if num_tokens == k:
            causal_mask = torch.triu(torch.ones((num_tokens, k), device=x.device, dtype=torch.bool), diagonal=1)
        else:
            offset = k - num_tokens
            row_idx = torch.arange(num_tokens, device=x.device).unsqueeze(1)
            col_idx = torch.arange(k, device=x.device).unsqueeze(0)
            causal_mask = row_idx + offset < col_idx

        attention_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -torch.inf)
        

        attention_weights = torch.softmax(attention_scores/keys.shape[-1]**0.5, dim=-1)
        attention_dropout = self.dropout(attention_weights)

        context_vectors = (attention_dropout @ values).transpose(1,2)

        context_vectors = context_vectors.contiguous().view(b, num_tokens, self.d_out)
        context_vectors = self.out_proj(context_vectors)

        return context_vectors

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None


class MHAScaledDotProduct_with_KV_Cache(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, max_seq_len=None, window_size=None):
        super().__init__()
        assert(d_out % num_heads) == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # kv cache
        self.max_seq_len = max_seq_len or context_length
        self.window_size = window_size or self.max_seq_len
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)


    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape

        if use_cache:
            assert num_tokens <= self.window_size, (f"input chunk size {num_tokens} exceeds the window size {self.window_size}")

        qkv = self.qkv(x)
        qkv = qkv.view(b, num_tokens, 3, self.num_heads, self.head_dim)

        #(3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv

        use_dropout = 0 if not self.training else self.dropout

        if use_cache:
            if self.cache_k is None or self.cache_k.size(0) != b:
                self.cache_k = torch.zeros((b,self.num_heads,self.window_size,self.head_dim), device=x.device)
                self.cache_v = torch.zeros_like(self.cache_k)
                self.ptr_cur = 0
            
            if self.ptr_cur + num_tokens > self.window_size:
                overflow = self.ptr_cur + num_tokens - self.window_size
                self.cache_k[:, :, :-overflow, :] = self.cache_k[:, :, overflow:, :].clone()
                self.cache_v[:, :, :-overflow, :] = self.cache_v[:, :, overflow:, :].clone()

            self.cache_k[:, :, self.ptr_cur:self.ptr_cur+num_tokens, :] = key
            self.cache_v[:, :, self.ptr_cur:self.ptr_cur+num_tokens, :] = value
            self.ptr_cur += num_tokens

            keys = self.cache_k[:, :, :self.ptr_cur, :]
            values = self.cache_v[:, :, :self.ptr_cur, :]

        else:
            keys, values = key, value
            self.ptr_cur = 0


        k = keys.size(-2)

        if num_tokens == k:
            causal_mask = torch.triu(torch.ones((num_tokens, k), device=x.device, dtype=torch.bool), diagonal=1)
            context_vectors = torch.nn.functional.scaled_dot_product_attention(query, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)
        else:
            offset = k - num_tokens
            row_idx = torch.arange(num_tokens, device=x.device).unsqueeze(1)
            col_idx = torch.arange(k, device=x.device).unsqueeze(0)
            causal_mask = row_idx + offset < col_idx
            context_vectors = torch.nn.functional.scaled_dot_product_attention(query, keys, values, attn_mask=causal_mask, dropout_p=use_dropout, is_causal=False)


        context_vectors = context_vectors.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        context_vectors = self.out_proj(context_vectors)

        return context_vectors

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None


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
        self.att = MultiHeadAttention(d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], context_length=cfg["context_length"], dropout=cfg["drop_rate"], num_heads=cfg["n_heads"], qkv_bias=cfg["qkv_bias"], window_size=cfg["kv_window_size"] if "kv_window_size" in cfg else cfg["context_length"])

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

class TransformerBlock_kv_cache_and_flashattn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MHAScaledDotProduct_with_KV_Cache(d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], context_length=cfg["context_length"], dropout=cfg["drop_rate"], num_heads=cfg["n_heads"], qkv_bias=cfg["qkv_bias"], window_size=cfg["kv_window_size"] if "kv_window_size" in cfg else cfg["context_length"])

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
        self.kv_window_size = cfg["kv_window_size"] if "kv_window_size" in cfg else cfg["context_length"]

    def forward(self, x, use_cache=False):
        batch_size, seq_len = x.shape
        tok_embeds = self.tok_emb(x)

        context_length = self.pos_emb.num_embeddings

        if use_cache:
            assert self.ptr_current_pos + seq_len <= context_length,( 
            f"position embedding overflow. want to read {self.ptr_current_pos + seq_len} which exceeded the size of {context_length}")

            pos_ids = torch.arange(self.ptr_current_pos, self.ptr_current_pos + seq_len, device=x.device, dtype=torch.long)
            self.ptr_current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=x.device, dtype=torch.long)

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

class GPTModel_with_KV_Cache_and_FlashAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList([TransformerBlock_kv_cache_and_flashattn(cfg) for _ in range(cfg["n_layers"])])
        self.ptr_current_pos = 0 
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.kv_window_size = cfg["kv_window_size"] if "kv_window_size" in cfg else cfg["context_length"]

    def forward(self, x, use_cache=False):
        batch_size, seq_len = x.shape
        tok_embeds = self.tok_emb(x)

        context_length = self.pos_emb.num_embeddings

        if use_cache:
            assert self.ptr_current_pos + seq_len <= context_length,( 
            f"position embedding overflow. want to read {self.ptr_current_pos + seq_len} which exceeded the size of {context_length}")

            pos_ids = torch.arange(self.ptr_current_pos, self.ptr_current_pos + seq_len, device=x.device, dtype=torch.long)
            self.ptr_current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=x.device, dtype=torch.long)

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
    kv_window_size = model.kv_window_size

    with torch.no_grad():
        if use_cache:
            input_tokens = idx[:, -ctx_length:]
            input_tokens_length = input_tokens.size(1)

            model.reset_kv_cache()

            for i in range(0, input_tokens_length, kv_window_size):
                idx = input_tokens[:, i:i+kv_window_size]
                logits = model(idx, use_cache)

            max_new_generable = ctx_length - model.ptr_current_pos
            max_new_tokens = min(max_new_generable, max_new_tokens)

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
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     
        "context_length": 1024,  
        "emb_dim": 768,          
        "n_heads": 12,           
        "n_layers": 12,          
        "drop_rate": 0.1,       
        "qkv_bias": False,   
        "kv_window_size":1024
    }


    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    out = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=200
    )

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)

    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes/(1024**3)
        print(f"maximum memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()