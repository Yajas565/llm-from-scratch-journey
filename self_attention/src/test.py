from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from EmbAndDataLoader.src.dataloader import GPTDatasetV1, create_dataloader_v1
from multihead_attention import MultiHeadAttention

text_path = REPO_ROOT / "tokenizer" / "the-verdict.txt"
with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()

data = create_dataloader_v1(
    text,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)

data = iter(data)
x ,y = next(data)

token_embedding = torch.nn.Embedding(50257, 768)
x = token_embedding(x)

mha = MultiHeadAttention(d_in=768, d_out=768, context_length=1024, dropout=0.1, num_heads=12)
out = mha(x)
print(out.shape)