import torch
from tokenizer_utils import tokenize_text
from bpe_tokenizer import get_bpe_tokenizer
from layer_normalization import LayerNormalization

words = "Tokenization is important in NLP."
token = tokenize_text(words)
print(token)

print("\n BPE Tokenizer")
bpe_tokens, bpe_ids = get_bpe_tokenizer()
print("BPE Tokens:", bpe_tokens)
print("BPE IDs:", bpe_ids)

input_data = torch.randn(10, 64)
model = LayerNormalization(64,32)
output = model(input_data)
print(output)