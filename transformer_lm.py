# models.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

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
    def __init__(self, d_model: int, d_attn: int, max_len: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_attn, bias=False)
        self.W_k = nn.Linear(d_model, d_attn, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.scale = math.sqrt(d_attn)
        # reuse causal mask to avoid re-alloc
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1),
            persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [seq len, d_model]
        :return: [seq len, d_model]
        """
        B, T, D = x.shape
        Q = self.W_q(x)  # [seq len, d_attn]
        K = self.W_k(x)  # [seq len, d_attn]
        V = self.W_v(x)  # [seq len, d_model]

        if x.device.type == "cuda":
            # fast path on GPU
            scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale   # [B,T,T]
            scores = scores.masked_fill(self.causal_mask[:T, :T], float('-inf'))
            scores = scores - scores.amax(dim=-1, keepdim=True)
            attn = F.softmax(scores, dim=-1)                           # [B,T,T]
            out  = torch.matmul(attn, V)                               # [B,T,D]
            return out  # no attn return
        
            
        outs = []
        KT = K.transpose(1, 2)     # [B, d_attn, T]

        for t in range(T):
            s = torch.matmul(Q[:, t:t+1, :], KT) / self.scale          # [B,1,T]
            # causal mask for this row
            if t + 1 < T:
                s[:, :, t+1:] = float('-inf')
            s = s - s.amax(dim=-1, keepdim=True) 
            a = F.softmax(s, dim=-1)                                   # [B,1,T]
            o = torch.matmul(a, V)                                     # [B,1,D]
            outs.append(o)
        out = torch.cat(outs, dim=1)                                   # [B,T,D]
        return out  # no attn return

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_attn: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, d_attn, max_len)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model * 4)
        self.ff2 = nn.Linear(d_model * 4, d_model)
        self.drop2 = nn.Dropout(dropout)   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self-attention + residual 
        h = self.attn(self.ln1(x))  # [seq len, d_model], [seq len, seq len]
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
                 n_layers: int = 2, dropout: float = 0.1,
                 use_checkpoint: bool = True
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.max_len = max_len
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, d_attn, max_len, dropout) for _ in range(n_layers)]
        )       
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.use_checkpoint = use_checkpoint

        # Xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        :param x: [seq len] LongTensor of token indices
        :return: [seq len, vocab size] log probabilities over next token at each position
        """
        if idx.dim() == 1:
            idx = idx.unsqueeze(0) 

        B, T = idx.size()
        
        # Token and positional embeddings
        pos = torch.arange(T, dtype=torch.long, device=idx.device).unsqueeze(0).expand(B, T)  
        x = self.token_emb(idx) + self.pos_emb(pos)  # [B, T, d_model]
        # Transformer blocks
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(lambda y: block(y), x) 
            else:
                x = block(x)  # [B, T, d_model]
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
        self.device = next(model.parameters()).device

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
            
            logits = self.model(context_idxs.unsqueeze(0))  # [context len, vocab size]
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

def _batch_iter(indices: np.ndarray, block_size: int, stride: int, batch_size: int, device: torch.device):
    """
    Stream (x,y) as [B,T] LongTensors already on `device`, without materializing all windows.
    """
    ids = torch.tensor(indices, dtype=torch.long, device=device)   # whole corpus on device once
    N = ids.numel()
    last_start = N - block_size - 1
    if last_start < 0:
        return

    starts = torch.arange(0, last_start + 1, step=stride, dtype=torch.long, device=device)  # [S]
    offsets = torch.arange(block_size, dtype=torch.long, device=device)                      # [T]
    offsets_next = offsets + 1

    for i in range(0, starts.numel(), batch_size):
        s = starts[i:i + batch_size]                                       # [B]
        idx_mat = s.unsqueeze(1) + offsets                                 # [B,T]
        x = ids.index_select(0, idx_mat.reshape(-1)).view(-1, block_size)  # [B,T]
        y = ids.index_select(0, (s.unsqueeze(1) + offsets_next).reshape(-1)).view(-1, block_size)
        yield x, y

def print_mem(tag: str):
    try:
        import psutil, os, gc
        gc.collect()
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
        print(f"[MEM] {tag}: {rss:.1f} MB")
    except Exception:
        pass

def train_lm(args, train_text, dev_text, vocab_index):
    # Hyperparameters
    block_size = getattr(args, 'block_size', 128)
    d_model    = getattr(args, 'd_model', 192)
    d_attn     = getattr(args, 'd_attn', 96)
    n_layers   = getattr(args, 'n_layers', 2)
    dropout    = getattr(args, 'dropout', 0.05)
    lr         = getattr(args, 'lr', 1e-4)
    batch_size = getattr(args, 'batch_size', 64)
    n_epochs   = getattr(args, 'n_epochs', 15)
    stride     = getattr(args, 'stride', 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        block_size = min(block_size, 112)    
        batch_size = min(batch_size, 16)
        n_layers   = min(n_layers, 2)
        d_model    = min(d_model, 192)
        d_attn     = min(d_attn, 96)

    # Build indices once
    train_indices = np.array(_build_corpus_indices(train_text, vocab_index), dtype=np.int64)
    # Edge case: very short corpus → pad
    if train_indices.size < block_size + 1:
        seed = np.array(_build_corpus_indices(" " + train_text, vocab_index), dtype=np.int64)
        while seed.size < block_size + 1:
            seed = np.tile(seed, 2)
        train_indices = seed[:block_size + 1]

    # Model
    model = TinyTransformerLM(
        vocab_size=len(vocab_index),
        max_len=block_size,
        d_model=d_model,
        d_attn=d_attn,
        n_layers=n_layers,
        dropout=dropout,
        use_checkpoint=True
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, eps = 1e-8, weight_decay=0.01, betas=(0.9, 0.95))
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    # print_mem("before epoch")
    for epoch in range(n_epochs):
        token_loss_sum = 0.0
        token_count    = 0

        for x, y in _batch_iter(train_indices, block_size, stride, batch_size, device):
            opt.zero_grad(set_to_none=True)
            # Forward
            logits = model(x)                         # [B,T,V]
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B*T, V), y.view(B*T))

            # Guard against NaN/Inf
            if not torch.isfinite(loss):
                print("Non-finite loss; skipping batch and reducing LR.")
                for pg in opt.param_groups:
                    pg['lr'] = max(pg['lr'] * 0.5, 1e-6)
                continue

             # Backward + clip + step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

             # Metrics
            token_loss_sum += float(loss.item()) * (B * T)
            token_count    += (B * T)
            # print_mem("after step")

        avg_ce = token_loss_sum / max(1, token_count)
        if not math.isfinite(avg_ce):
            ppl = float('inf')
        else:
            ppl = math.exp(min(avg_ce, 80.0))
        print(f"Epoch {epoch+1}/{n_epochs} — CE {avg_ce:.4f} — PPL {ppl:.2f}")

    model.eval()
    lm = NeuralLanguageModel(model, vocab_index)  # shares device with model
    return lm
