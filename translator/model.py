from typing import Optional, Any

import torch
from torch import nn, Tensor

from translator.generic_classes import (
    EmbeddingLayer,
    PositionalEncodingLayer,
    MultiHeadAttention,
    NormalizeLayer,
    FeedForwardLayer
)

from translator.constants import MASK_TOKEN


class EncoderLayer(nn.Module):
    def __init__(
            self,
            embeddings_len: int,
            nb_heads: int,
            feedforward_len: int,
            dropout=0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(nb_heads, embeddings_len)
        self.normalize = NormalizeLayer(embeddings_len)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = FeedForwardLayer(embeddings_len, feedforward_len, dropout)

    def forward(self, tensor: Tensor, encoder_mask: Optional[Tensor]) -> Tensor:
        temp_tensor = tensor + self.dropout(self.attention(tensor, tensor, tensor, encoder_mask))
        temp_tensor = self.normalize(temp_tensor)
        temp_tensor = temp_tensor + self.dropout(self.feed_forward(tensor))
        return self.normalize(temp_tensor)


class Encoder(nn.Module):
    def __init__(
            self,
            nb_layer: int,
            vocabulary_size: int,
            embeddings_len: int,
            nb_heads: int,
            feedforward_len: int,
            device: Any,
            dropout: float = 0.1
    ):
        super().__init__()
        self.device = device
        self.embeddings_len = embeddings_len

        self.seq_embedding_layer = EmbeddingLayer(vocabulary_size, embeddings_len)
        self.positional_encoder = PositionalEncodingLayer(embeddings_len)
        self.layers = nn.ModuleList([
            EncoderLayer(embeddings_len, nb_heads, feedforward_len, dropout)
            for _ in range(nb_layer)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_word_ids: Tensor, encoder_mask) -> Tensor:
        batch_size = input_word_ids.size(0)
        seq_with_pad_len = input_word_ids.size(1)
        pos_embeddings = self.positional_encoder(batch_size, seq_with_pad_len).to(self.device)
        seq_embedding_layer = self.seq_embedding_layer(input_word_ids).to(self.device)

        scale = torch.sqrt(torch.FloatTensor([self.embeddings_len])).to(self.device)

        input_tensor = pos_embeddings + seq_embedding_layer * scale

        input_tensor = self.dropout(input_tensor)
        for layer in self.layers:
            input_tensor = layer(input_tensor, encoder_mask)
        return input_tensor


class DecoderLayer(nn.Module):
    def __init__(
            self,
            embeddings_len: int,
            nb_heads: int,
            feedforward_len: int,
            dropout=0.1
    ):
        super().__init__()
        self.enc_dec_attention = MultiHeadAttention(nb_heads, embeddings_len)
        self.mask_attention = MultiHeadAttention(nb_heads, embeddings_len)
        self.normalize = NormalizeLayer(embeddings_len)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = FeedForwardLayer(embeddings_len, feedforward_len, dropout)

    def forward(self, tensor: Tensor, hidden_state: Tensor, decoder_mask: Optional[Tensor]) -> Tensor:
        temp_tensor = tensor + self.dropout(self.mask_attention(tensor, tensor, tensor, decoder_mask))
        temp_tensor = self.normalize(temp_tensor)
        temp_tensor = temp_tensor + self.dropout(self.enc_dec_attention(temp_tensor, hidden_state, hidden_state, decoder_mask))
        temp_tensor = self.normalize(temp_tensor)
        temp_tensor = temp_tensor + self.dropout(self.feed_forward(tensor))
        return self.normalize(temp_tensor)


class Decoder(nn.Module):
    def __init__(
            self,
            nb_layer: int,
            vocabulary_size: int,
            embeddings_len: int,
            nb_heads: int,
            feedforward_len: int,
            device: Any,
            dropout: float = 0.1
    ):
        super().__init__()
        self.device = device
        self.embeddings_len = embeddings_len

        self.seq_embedding_layer = EmbeddingLayer(vocabulary_size, embeddings_len)
        self.positional_encoder = PositionalEncodingLayer(embeddings_len)
        self.layers = nn.ModuleList([
            DecoderLayer(embeddings_len, nb_heads, feedforward_len, dropout)
            for _ in range(nb_layer)])
        self.dropout = nn.Dropout(dropout)
        self.out_vocab_mapper = nn.Linear(embeddings_len, vocabulary_size)
        self.softmax_layer = nn.LogSoftmax(dim=-1)

    def forward(self, output_word_ids: Tensor, hidden_state: Tensor, decoder_mask: Optional[Tensor]) -> Tensor:
        batch_size = output_word_ids.size(0)
        seq_with_pad_len = output_word_ids.size(1)

        pos_embeddings = self.positional_encoder(batch_size, seq_with_pad_len).to(self.device)
        seq_embedding_layer = self.seq_embedding_layer(output_word_ids).to(self.device)

        scale = torch.sqrt(torch.FloatTensor([self.embeddings_len])).to(self.device)
        output_tensor = pos_embeddings + seq_embedding_layer * scale
        output_tensor = self.dropout(output_tensor)

        for layer in self.layers:
            output_tensor = layer(output_tensor, hidden_state, decoder_mask)

        # return self.softmax_layer(self.out_vocab_mapper(output_tensor))
        return self.out_vocab_mapper(output_tensor)


def create_src_mask(src_input_ids: Tensor, default_mask: Optional[Tensor], device) -> Tensor:
    if default_mask is not None:
        return default_mask.unsqueeze(1)

    return (src_input_ids != MASK_TOKEN).unsqueeze(1).tode(device)


def create_trg_mask(trg_input_ids: Tensor, default_mask: Optional[Tensor], device) -> Tensor:
    if default_mask is None:
        default_mask = (trg_input_ids != MASK_TOKEN).unsqueeze(1)

    default_mask = default_mask.unsqueeze(1)

    trg_seq_len = trg_input_ids.shape[1]
    # Create a triangle matrix with False in upper-right
    trg_bool_mask = torch.tril(torch.ones((trg_seq_len, trg_seq_len))).bool().to(device)
    return default_mask & trg_bool_mask


class TransformerModel(nn.Module):
    def __init__(
            self,
            encoder_layer: int,
            decoder_layer: int,
            encoder_heads: int,
            decoder_heads: int,
            input_vocab_size: int,
            embeddings_len: int,
            feedforward_len: int,
            target_vocab_size: int,
            device: Any,
            dropout=0.1
    ):
        super().__init__()
        self.encoder = Encoder(
            encoder_layer,
            input_vocab_size,
            embeddings_len,
            encoder_heads,
            feedforward_len,
            device,
            dropout
        )

        self.decoder = Decoder(
            decoder_layer,
            target_vocab_size,
            embeddings_len,
            decoder_heads,
            feedforward_len,
            device,
            dropout
        )
        self.device = device
        self.out_vocab_mapper = nn.Linear(embeddings_len, target_vocab_size)

    def forward(
            self,
            input_ids: Tensor,
            trg_input_ids: Tensor,
            attention_mask: Optional[Tensor],
            trg_mask: Optional[Tensor]
    ) -> Tensor:
        """
        This function takes the source tensor and the target one and returns logit distribution over the target vocabulary items.
        @param trg_mask:
        @param attention_mask:
        @param input_ids:
        @param trg_input_ids:
        @return:
        """
        encoder_mask = create_src_mask(input_ids, attention_mask, self.device)
        decoder_mask = create_trg_mask(trg_input_ids, trg_mask, self.device)

        hidden_states = self.encoder(input_ids, encoder_mask)
        decoder_output = self.decoder(trg_input_ids, hidden_states, decoder_mask)
        return decoder_output

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# src_input_ids = torch.randint(0, INPUT_VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
# src_mask = torch.randint(0, INPUT_VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
#
# trg_input_ids = torch.randint(0, OUTPUT_VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
# trg_mask = torch.randint(0, OUTPUT_VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
#
# transformers = TransformerModel(
#     encoder_layer=2,
#     decoder_layer=2,
#     encoder_heads=2,
#     decoder_heads=2,
#     input_vocab_size=INPUT_VOCAB_SIZE,
#     embeddings_len=512,
#     feedforward_len=1024,
#     max_seq_len=MAX_SEQ_LEN,
#     target_vocab_size=OUTPUT_VOCAB_SIZE,
#     batch_size=BATCH_SIZE,
#     device=device,
#     dropout=0.1
# )

# print(device)
# out = transformers(src_input_ids, src_mask, trg_input_ids, trg_mask)
# print(out)
