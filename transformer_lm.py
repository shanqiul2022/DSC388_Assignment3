# models.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")

class CausalSelfAttention(nn.Module):
    """
    Single-headed causal self-attention layer.
    - Query, Key, Value are all computed in d_attn
    - Values in d_model
    - Uses a causal mask to prevent attention to future positions
    """
    def __init__(self, d_model: int, d_attn: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_attn, bias=False)
        self.W_k = nn.Linear(d_model, d_attn, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.scale = math.sqrt(d_attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [seq len, d_model]
        :return: [seq len, d_model]
        """
        B, T, D = x.shape
        Q = self.W_q(x)  # [seq len, d_attn]
        K = self.W_k(x)  # [seq len, d_attn]
        V = self.W_v(x)  # [seq len, d_model]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale  # [seq len, seq len]

        # Causal mask to prevent attending to future positions
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device = x.device), diagonal=1)  # [seq len, seq len]
        scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # [seq len, seq len]
        attn_output = torch.matmul(attn_weights, V)  # [seq len, d_model]

        return attn_output, attn_weights  # return attention weights for visualization

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_attn: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, d_attn)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model * 4)
        self.ff2 = nn.Linear(d_model * 4, d_model)
        self.drop2 = nn.Dropout(dropout)   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self-attention + residual 
        h, _ = self.attn(self.ln1(x))  # [seq len, d_model], [seq len, seq len]
        x = x + self.drop1(h)  # [seq len, d_model]

        # position wise feedforward + residual
        z = self.ff2(F.relu(self.ff1(self.ln2(x))))  # [seq len, d_model]
        x = x + self.drop2(z)  # [seq len, d_model]
        return x  # return attention weights for visualization

class TinyTransformerLM(nn.Module):
    
    """
    A tiny transformer language model with
    - token embeddings
    - positional embeddings
    - N causal Transformer blocks
    - Linear head to vocab
    Expect input as a 1D LongTensor of token indices [T]
    """

    def __init__(self, vocab_size: int, max_len: int,
                 d_model: int = 128, d_attn: int = 64, 
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.max_len = max_len
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_attn, dropout) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # Xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        :param x: [seq len] LongTensor of token indices
        :return: [seq len, vocab size] log probabilities over next token at each position
        """
        B, T = idx.size()
        
        # Token and positional embeddings
        pos = torch.arange(T, dtype=torch.long, device=idx.device).unsqueeze(0).expand(B, T)   # [seq len]
        x = self.token_emb(idx) + self.pos_emb(pos)  # [seq len, d_model]
        # Transformer blocks
        for block in self.blocks:
            x = block(x)  # [seq len, d_model]
        # Language modeling head
        logits = self.lm_head(x)  # [seq len, vocab size]
        return logits

class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model: TinyTransformerLM, vocab_index):
        super().__init__()
        self.model = model
        self.vocab_index = vocab_index
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _string_to_indices(self, s):
        # Map each character to index; assume vocab is complete (27 chars)
        idxs = [self.vocab_index.index_of(c) for c in s]
        return torch.tensor(idxs, dtype=torch.long, device=self.device)
    
    def get_next_char_log_probs(self, context):
        self.model.eval()  # set to eval mode to disable dropout
        with torch.no_grad():
            context_idxs = self._string_to_indices(context)  # [context len]
            if context_idxs.size(0) > self.model.max_len:
                context_idxs = context_idxs[-self.model.max_len:]  # truncate to max length
            if context_idxs.size(0) == 0:
                # empty context, use a single space
                context_idxs = self._string_to_indices(" ")
            
            logits = self.model(context_idxs)  # [context len, vocab size]
            last_logits = logits[0, -1, :]  # [vocab size]    
            log_probs = F.log_softmax(last_logits, dim=-1)  # [context len, vocab size]
            return log_probs.cpu().numpy()  # return log probs for the last position

    def get_log_prob_sequence(self, next_chars, context):
        self.model.eval()  # set to eval mode to disable dropout
        total_log_prob = 0.0
        running = context
        with torch.no_grad():
            for c in next_chars:
                lp = self.get_next_char_log_probs(running)  # [vocab size]
                idx = self.vocab_index.index_of(c)
                total_log_prob += float(lp[idx])
                running += c

        return total_log_prob

def _build_corpus_indices(text, vocab_index):
    """
    Convert text to a list of indices using the given vocab_index
    :param text: input text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a list of indices
    """
    indices = [vocab_index.index_of(c) for c in text]
    return indices

def _make_lm_batches(indices: np.ndarray, block_size: int, step:int):
    """
    Convert a list of indices into batches for training the LM.
    Each batch is a tuple (input, target) where input and target are LongTensors of shape [batch size, seq len].
    The target is the input shifted by one character.
    :param indices: a list of indices
    :param batch_size: the batch size
    :param seq_len: the sequence length
    :return: a list of (input, target) tuples
    """
    N = len(indices)
    xs, ys = [], []
    last_start = N - block_size - 1
    for start in range(0, last_start+1, step):
        x = indices[start:start+block_size]
        y = indices[start+1:start+block_size+1]
        if len(x) == block_size and len(y) == block_size:
            xs.append(x)
            ys.append(y)
    if not xs:
        return None, None
    x_tensor = torch.tensor(np.stack(xs), dtype=torch.long)
    y_tensor = torch.tensor(np.stack(ys), dtype=torch.long)
    return x_tensor, y_tensor

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # Hyperparameters
    block_size = getattr(args, 'block_size', 256)  # sequence length
    d_model = getattr(args, 'd_model', 256)
    d_attn = getattr(args, 'd_attn', 128)
    n_layers = getattr(args, 'n_layers', 3)
    dropout = getattr(args, 'dropout', 0.05)
    lr = getattr(args, 'lr', 3e-4)
    batch_size = getattr(args, 'batch_size', 64)
    n_epochs = getattr(args, 'n_epochs', 5)
    stride = getattr(args, 'stride', 1)

    # Read data
    train_indices = _build_corpus_indices(train_text, vocab_index)  # list of indices
    X_train, Y_train = _make_lm_batches(np.array(train_indices), block_size, stride)
    if X_train is None:
        seq = _build_corpus_indices(" " + train_text, vocab_index)
        if len(seq) < block_size + 1:
            pad = np.tile(seq, math.ceil((block_size + 1) / len(seq)))
            seq = pad[:block_size + 1]
        X_train = torch.tensor([seq[:block_size]], dtype=torch.long)
        Y_train = torch.tensor([seq[1:block_size + 1]], dtype=torch.long)
    
    # Shuffle training data
    perm = torch.randperm(X_train.size(0))
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    model = TinyTransformerLM(
        vocab_size=len(vocab_index),
        max_len=block_size,
        d_model=d_model,
        d_attn=d_attn,
        n_layers=n_layers,
        dropout=dropout,
    )
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        num_tokens = 0

        for i in range(0, X_train.size(0), batch_size):
            x = X_train[i:i+batch_size].to(device)  # [batch size, seq len]
            y = Y_train[i:i+batch_size].to(device)  # [batch size, seq len]
            
            opt.zero_grad()
            logits = model(x)

            B, T, V = logits.shape
            loss = loss_fn(logits.reshape(B*T, V), y.reshape(B*T))
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
    
    model.eval()
    lm = NeuralLanguageModel(model, vocab_index)
    return lm
