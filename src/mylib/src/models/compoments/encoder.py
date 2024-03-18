import torch
from torch import nn
from torch.nn.functional import one_hot

class TransformerEncoder(nn.Module):
    def __init__(self, look_back, cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, emb_dim):
        super(TransformerEncoder, self).__init__()

        self.look_back = look_back
        self.cat_vocab_size = cat_vocab_size
        self.amount_vocab_size = amount_vocab_size
        self.emb_dim = emb_dim

        self.id_embedding = nn.Embedding(num_embeddings=id_vocab_size, embedding_dim=3*emb_dim)
        self.cat_embedding = nn.Embedding(num_embeddings=cat_vocab_size, embedding_dim=emb_dim)
        self.amount_embedding = nn.Embedding(num_embeddings=amount_vocab_size+1, embedding_dim=emb_dim,
                                             padding_idx=amount_vocab_size)
        self.dt_embedding = nn.Embedding(num_embeddings=dt_vocab_size, embedding_dim=emb_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=look_back, embedding_dim=emb_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=3*emb_dim, nhead=4, dim_feedforward=3*emb_dim,
                                                        dropout=0.2, activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    def forward(self, cat_arr, dt_arr, amount_arr, id_arr):
        x_id_emb = self.id_embedding(id_arr).unsqueeze(1)  # [batch_size, 1, emb_dim]
        x_cat_emb = self.cat_embedding(torch.arange(self.cat_vocab_size)).unsqueeze(0).expand(cat_arr.shape[0], -1, -1)
        # [batch_size, cat_vocab_size, emb_dim]
        x_mask_cat = torch.tensor(~(cat_arr == self.cat_vocab_size), dtype=torch.int64).unsqueeze(3)
        # [batch_size, look_back, max_cat_len, 1]
        x_onehot_cat = torch.sum(one_hot(cat_arr, num_classes=self.cat_vocab_size+1) * x_mask_cat, dim=2)[:, :, :-1]
        # [batch_size, look_back, cat_vocab_size]

        x_amount_emb = torch.sum(self.amount_embedding(x_onehot_cat.index_put_(indices=torch.where(x_onehot_cat == 0),
                                                       values=torch.tensor(self.amount_vocab_size)).index_put_(indices=torch.where(x_onehot_cat == 1),
                                                       values=amount_arr.flatten()[amount_arr.flatten() != self.amount_vocab_size])) *
                                                       x_onehot_cat.unsqueeze(3), dim=1)
        # [batch_size, cat_vocab_size, emb_dim]

        x_dt_emb = torch.sum((self.dt_embedding(dt_arr) +
                              self.pos_embedding(torch.arange(dt_arr.shape[1])).unsqueeze(0).expand(dt_arr.shape[0], -1, -1)).unsqueeze(2).expand(-1, -1, self.cat_vocab_size, -1) *
                              x_onehot_cat.unsqueeze(3), dim=1)

        x_encoder_input = torch.cat([x_id_emb, torch.cat([x_cat_emb, x_dt_emb, x_amount_emb], dim=2)], dim=1)
        # [batch_size, cat_vocab_size+1, emb_dim]

        x_encoder_output = self.transformer_encoder(x_encoder_input)[:, 1:, :]
        # [batch_size, cat_vocab_size, 3*emb_dim]
        return x_encoder_output
