from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_bpe_tokenizer():

    trainer = BpeTrainer(vocab_size=100) 

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(["sample/input.txt"], trainer)

    output = tokenizer.encode("This is a sample input text to test the BPE tokenizer.")

    return output.tokens, output.ids