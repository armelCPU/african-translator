import math
from typing import Optional

import torch
import torch.nn.functional as f
from torch import nn, Tensor

from translator.constants import (
    EPSILON,
    DROPOUT,
    MASK_TOKEN,
    MASK_VALUE
)


class EmbeddingLayer(nn.Module):
    """
    Given a word in a sentence, this layer return its embeddings.
    These embeddings summaries word information and have a fixed size whatever is given as input.

    """
    def __init__(self, vocabulary_len: int, embeddings_len: int):
        super().__init__()
        self.emb = nn.Embedding(vocabulary_len, embeddings_len)

    def forward(self, words_ids: Tensor):
        return self.emb(words_ids)


class PositionalEncodingLayer(nn.Module):
    """
    Build positional embeddings for a text sequence.
    """
    def __init__(self, embeddings_len: int):
        super().__init__()
        self.embeddings_len = embeddings_len

    def forward(self, batch_size: int, max_sent_len: int) -> Tensor:
        positional_vect = torch.zeros(batch_size, max_sent_len, self.embeddings_len)

        for batch_item in range(batch_size):
            for word_pos in range(max_sent_len):
                for i in range(self.embeddings_len):
                    if i%2 == 0:
                        positional_vect[batch_item, word_pos, i] = math.sin(word_pos / (10000 ** ((2 * i)/self.embeddings_len)))
                    else:
                        positional_vect[batch_item, word_pos, i] = math.cos(word_pos / (10000 ** ((2 * (i + 1))/self.embeddings_len)))

        return positional_vect


class AttentionHead(nn.Module):
    def __init__(self, embeddings_len: int, query_len: int, key_len: int, value_len: int):
        super().__init__()
        # Apply different linear transformation to Q, K and V in this head
        self.query_layer = nn.Linear(embeddings_len, query_len)
        self.key_layer = nn.Linear(embeddings_len, key_len)
        self.value_layer = nn.Linear(embeddings_len, value_len)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor]) -> Tensor:
        head_query = self.query_layer(query)
        head_key = self.key_layer(key)
        head_value = self.value_layer(value)
        q_emb_len = head_query.size(-1)
        energy = head_query.bmm(head_key.transpose(2, 1))

        if mask is not None:
            energy = energy.masked_fill(mask == MASK_TOKEN, MASK_VALUE)

        att_matrix = f.softmax(energy/q_emb_len**0.5, dim=-1)
        return att_matrix.bmm(head_value)


class MultiHeadAttention(nn.Module):
    def __init__(self, nb_heads: int, embeddings_len: int):
        if embeddings_len%nb_heads != 0:
            raise Exception("Embeddings length must be a multiple of number of heads ...")

        super().__init__()

        # Compute the emb len of items in each attention head
        head_query_len = head_key_len = head_value_len = embeddings_len // nb_heads
        self.attention_heads = nn.ModuleList(
            [AttentionHead(embeddings_len, head_query_len, head_key_len, head_value_len)
             for _ in range(nb_heads)
             ]
        )
        self.linear_layer = nn.Linear(embeddings_len, embeddings_len)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor]) -> Tensor:
        scaled_dot_attentions = [attention_head(query, key, value, mask) for attention_head in self.attention_heads]
        # Concat tensors only on the last dimension (embeddings)
        head_attention_concat = torch.cat(scaled_dot_attentions, dim=-1)
        return self.linear_layer(head_attention_concat)


class FeedForwardLayer(nn.Module):
    def __init__(self, embeddings_len: int, feedforward_len: int, dropout: float = DROPOUT):
        super().__init__()
        self.linear_1 = nn.Linear(embeddings_len, feedforward_len)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(feedforward_len, embeddings_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tensor: Tensor) -> Tensor:
        temp_tensor = self.linear_1(tensor)
        temp_tensor = self.relu(temp_tensor)
        temp_tensor = self.dropout(temp_tensor)
        return self.linear_2(temp_tensor)


class NormalizeLayer(nn.Module):
    def __init__(self, embeddings_len, epsilon = EPSILON):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(embeddings_len))
        self.bias = nn.Parameter(torch.zeros(embeddings_len))
        self.epsilon = epsilon

    def forward(self, tensor: Tensor) -> Tensor:
        return self.alpha * (tensor - tensor.mean(dim=-1, keepdim=True)) \
               / (tensor.std(dim=-1, keepdim=True) + self.epsilon) + self.bias

