import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from tqdm import tqdm

with open('shakespeare.txt') as f:
    text = f.read()
dictionary = {}
for character in text:
    if character in dictionary:
        dictionary[character] += 1
    else:
        dictionary[character] = 1
character_to_index = {}
index_to_character = {index: character for index, character in enumerate(dictionary.keys())}
character_to_index = {character: index for index, character in enumerate(dictionary.keys())}
vocab_size = len(dictionary.keys())
embedding_size = 64
block_size = 64
n_layers = 6
n_head = 8
device = 'cuda:0'

def tokenize(sequence_of_characters):
    tokens = []
    for character in sequence_of_characters:
        tokens.append(character_to_index[character])
    return tokens

def decode(sequence_of_tokens):
    characters = []
    for token in sequence_of_tokens:
        characters.append(index_to_character[token])
    return characters

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.linear = nn.Linear(embedding_size, 3 * embedding_size)
        self.linear2 = nn.Linear(embedding_size, embedding_size)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
    def forward(self, x):
        batch_size, sequence_length, embedding_size = x.size()
        qkv = self.linear(x) # batched qkv computation
        q, k, v = qkv.split(embedding_size, dim=2)
        q = q.view(batch_size, sequence_length, n_head, embedding_size // n_head).transpose(1, 2)
        k = k.view(batch_size, sequence_length, n_head, embedding_size // n_head).transpose(1, 2)
        v = v.view(batch_size, sequence_length, n_head, embedding_size // n_head).transpose(1, 2)
        matmul = q @ k.transpose(-2, -1)
        scale = matmul * (1.0 / math.sqrt(k.size(-1)))
        mask = scale.masked_fill(self.bias[:,:,:sequence_length,:sequence_length] == 0, float('-inf'))
        softmax = F.softmax(mask, dim=-1)
        matmul = softmax @ v
        concat = matmul.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_size)
        linear = self.linear2(concat)
        return linear

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.attention = Attention()
        self.layer_norm_1 = nn.LayerNorm(embedding_size)
        self.feed_forward = nn.Linear(embedding_size, embedding_size)
        self.layer_norm_2 = nn.LayerNorm(embedding_size)
    def forward(self, x):
        x = self.attention(x) + x
        x = self.layer_norm_1(x)
        x = self.feed_forward(x) + x
        x = self.layer_norm_2(x)
        return x

class GPT(nn.Module):
    def __init__(self):
        super(GPT, self).__init__()
        self.input_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.positional_encoding = nn.Embedding(num_embeddings=block_size, embedding_dim=embedding_size)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.linear = nn.Linear(embedding_size, vocab_size)
    def forward(self, indices, targets=None):
        device = indices.device
        batch_size, sequence_length = indices.size()
        assert sequence_length <= block_size
        position = torch.arange(0, sequence_length, dtype=torch.long, device=device).unsqueeze(0)
        token_embeddings = self.input_embeddings(indices)
        positional_embeddings = self.positional_encoding(position)
        x = token_embeddings + positional_embeddings
        for block in self.blocks:
            x = block(x)
        if targets is not None:
            logits = self.linear(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.linear(x[:, [-1], :])
            loss = None
        return logits, loss

model = GPT().to(device)
print([tokenize(text[0:10])])    
print(model(torch.LongTensor([tokenize(text[0:10]), tokenize(text[10:20])]).to(device)))
batch_size = 16
optimizer = torch.optim.AdamW(model.parameters()) 
prompt = " "
sample_length = block_size - len(prompt)

for epoch in range(10):
    print('epoch:', epoch)
    index = 0
    losses = []
    for batch in tqdm(range(0, len(text) - batch_size * block_size, batch_size * block_size)):
        training_examples = []
        targets = []
        for _ in range(batch_size):
            training_example = text[index:index+block_size]
            target = text[index+1:index+1+block_size]
            index += block_size
            training_examples.append(tokenize(training_example))
            targets.append(tokenize(target))
        x = torch.LongTensor(training_examples).to(device)
        y = torch.LongTensor(targets).to(device)
        logits, loss = model(x, y)
        losses.append(loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print('mean loss:', np.mean(losses))
    completion = prompt
    print('prompt:', prompt)
    for _ in range(sample_length):
        x = torch.LongTensor([tokenize(completion)]).to(device)
        logits = model(x)
        completion += decode([np.argmax(logits[0][0].cpu().detach().numpy())])[0]
    print('completion:', completion)
