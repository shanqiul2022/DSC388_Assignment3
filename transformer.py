# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.posenc = PositionalEncoding(d_model, num_positions, batched=False)
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)
        self.num_positions = num_positions

        # small Xavier init to  help training
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        x = self.embed(indices)   # [seq len, embedding dim]
        x = self.posenc(x)
    
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)          # [seq len, embedding dim], [seq len, seq len]
            attn_maps.append(attn.detach()) # store for visualization later

        logits = self.classifier(x)  # [seq len, num classes]
        log_probs = nn.functional.log_softmax(logits, dim=-1)  # [seq len, num classes]
        return log_probs, attn_maps
    

# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        # single head self-attention projection matrices
        self.W_Q = nn.Linear(d_model, d_internal, bias=False)
        self.W_K = nn.Linear(d_model, d_internal, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias = False)  # value has same dim

        # position-wise feedforward layer
        d_ff = 4 * d_model  # typically 4x the model dimension
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

        # dropout for a touch of regularization
        self.dropout_att = nn.Dropout(p=0.1)
        self.dropout_ff = nn.Dropout(p=0.1)

        self.sqrt_dk = np.sqrt(d_internal)

    def forward(self, input_vecs):
        # compute Q, K, V
        Q = self.W_Q(input_vecs)  # [seq len, d_internal]
        K = self.W_K(input_vecs)  # [seq len, d_internal]
        V = self.W_V(input_vecs)  # [seq len, d_model]

        # Attention scores and weights
        attn_scores = torch.matmul(Q, K.transpose(0, 1)) / self.sqrt_dk  # [seq len, seq len]
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)  # [seq len, seq len]
        attn_weights = self.dropout_att(attn_weights)

        # weighted sum of values
        context = torch.matmul(attn_weights, V)  # [seq len, d_model]

        # residual 1
        x = input_vecs + context  # [seq len, d_model]

        # position-wise feedforward + residual 2
        ff_out = self.ff2(nn.functional.relu(self.ff1(x)))  # [seq len, d_model]
        ff_out = self.dropout_ff(ff_out)
        output_vecs = x + ff_out  # [seq len, d_model]      

        return output_vecs, attn_weights  # [seq len, d_model], [seq len, seq len]


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    
    # hyperparameters 
    d_model = 64
    d_internal = 32
    num_layers = 2
    num_classes = 3
    vocab_size = 27
    seq_len = 20
  
    # The following code DOES NOT WORK but can be a starting point for your implementation
    # Some suggested snippets to use:
    model = Transformer(
                        vocab_size= vocab_size,
                        num_positions= seq_len,
                        d_model= d_model,
                        d_internal= d_internal,
                        num_classes= num_classes,
                        num_layers= num_layers,
                        )
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fcn = nn.NLLLoss()

    num_epochs = 10
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        ex_idxs = list(range(0, len(train)))
        random.shuffle(ex_idxs)

        for ex_idx in ex_idxs:
            ex = train[ex_idx]
            log_probs, _ = model.forward(ex.input_tensor)
            loss = loss_fcn(log_probs, ex.output_tensor) 

            optimizer.zero_grad()   # clear gradients for this training step
            loss.backward()         # backpropagate the loss
            optimizer.step()        # update the parameters

            loss_this_epoch += loss.item()

    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
