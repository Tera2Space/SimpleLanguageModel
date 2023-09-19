from torch import nn
from torch.functional import F
import torch

class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden, layers, block_size, head_size, device):
        super().__init__()
        self.device = device
        self.head_size = head_size
        
        self.token_embedding_table = nn.Embedding(vocab_size, hidden)
        self.pos_embedding_table = nn.Embedding(block_size, hidden)
        
        self.key = nn.Linear(hidden, head_size, bias=False)
        self.query = nn.Linear(hidden, head_size, bias=False)
        self.value = nn.Linear(hidden, head_size, bias=False)
        
        hidden_layers = []
        for _ in range(layers):
            hidden_layers.append(nn.Linear(hidden,hidden))
            hidden_layers.append(nn.LeakyReLU())
            
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        self.lm_head = nn.Linear(hidden, vocab_size)
        
    def self_attention(self, x):
        _, T, _ = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        
        wei = query @ key.transpose(-2, -1) *  self.head_size ** -0.5# (B, T, head_size) @ (B, 16, T) ---> (B, T, T)
        
        trill = torch.tril(torch.ones((T,T),device=self.device))
        wei = wei.masked_fill(trill==0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = wei.to(self.device)
        # xbow = wei @ x
        xbow = wei @ value
        
        
        return xbow
    
    def forward(self, idx, targets=None):
        x = self.token_embedding_table(idx) # (B,T,C)
        
        B, T, C = x.shape

        pos_emb = self.pos_embedding_table(torch.arange(T, device=self.device))
        x += pos_emb
        
        x = self.self_attention(x)
        
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        
        x = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = x.shape
            x = x.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(x, targets)

        return x, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
