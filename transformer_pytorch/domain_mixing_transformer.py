import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 n_domain,
                 domain_eps,
                 device,
                 max_length=100
                 ):
        super().__init__()
        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim).to(device)
        self.pos_embedding = nn.Embedding(max_length, hid_dim).to(device)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, n_domain, domain_eps, device) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout).to(device)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch_size, src_len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        # pos = [batch_size, src_len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # src = [batch_size, src_len, hid_dim]
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        for i, layer in enumerate(self.layers):
            if i == 0:
                src, domain = layer(src, src_mask)
            else:
                src, d = layer(src, src_mask)
                domain += d

        return src, domain/len(self.layers)


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 n_domain,
                 domain_eps,
                 device
                 ):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, n_domain, domain_eps, device)
        self.position_wise_feedforward = PositionWiseFeedforwardLayer(hid_dim, pf_dim, dropout, n_domain, domain_eps, device)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch_size, src_len, hid_dim]

        _src, _, att_domain = self.self_attention(src, src, src, src_mask)

        src = self.self_attn_layer_norm(src + self.dropout(_src))

        _src, ff_domain = self.position_wise_feedforward(src)

        src = self.ff_layer_norm(src + self.dropout(_src))

        return src, (att_domain + ff_domain)/2


class PositionWiseFeedforwardLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 pf_dim,
                 dropout,
                 n_domain,
                 domain_eps,
                 device,
                 ):
        super().__init__()
        self.hid_dim = hid_dim
        self.device = device
        self.n_domain = n_domain
        self.domain_eps = domain_eps

        self.fc_r = nn.Linear(hid_dim, n_domain).to(device)

        self.fc_1 = [nn.Linear(hid_dim, pf_dim).to(device) for _ in range(n_domain)]
        self.fc_2 = [nn.Linear(pf_dim, hid_dim).to(device) for _ in range(n_domain)]

        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x):

        # d = [batch_size, seq_len, n_domain]
        d = (1 - self.domain_eps) * torch.softmax(self.fc_r(x), dim=-1) + self.domain_eps / self.n_domain

        x_out = torch.zeros(x.shape).to(self.device)

        for i_d in range(self.n_domain):
            x_i = self.dropout(torch.relu(self.fc_1[i_d](x)))
            x_i = self.fc_2[i_d](x_i)
            # x = [batch_size, seq_len, hid_dim]
            i_d = torch.repeat_interleave(d[:, :, i_d], self.hid_dim, dim=1).view(x_i.shape)
            x_out += (x_i*i_d)
        return x_out, d


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 dropout,
                 n_domain,
                 domain_eps,
                 device
                 ):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.device = device
        self.n_domain = n_domain
        self.domain_eps = domain_eps
        self.fc_rq = nn.Linear(hid_dim, n_domain).to(device)
        self.fc_rk = nn.Linear(hid_dim, n_domain).to(device)
        self.fc_rv = nn.Linear(hid_dim, n_domain).to(device)

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = [nn.Linear(hid_dim, hid_dim).to(device) for _ in range(n_domain)]
        self.fc_k = [nn.Linear(hid_dim, hid_dim).to(device) for _ in range(n_domain)]
        self.fc_v = [nn.Linear(hid_dim, hid_dim).to(device) for _ in range(n_domain)]

        self.fc_o = nn.Linear(hid_dim, hid_dim).to(device)

        self.dropout = nn.Dropout(dropout).to(device)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        # [batch_size, query_len, n_domain]
        dq = (1 - self.domain_eps) * torch.softmax(self.fc_rq(query), dim=-1) + self.domain_eps / self.n_domain
        q = torch.zeros(query.shape).to(self.device)
        dk = (1 - self.domain_eps) * torch.softmax(self.fc_rk(key), dim=-1) + self.domain_eps / self.n_domain
        k = torch.zeros(key.shape).to(self.device)
        dv = (1 - self.domain_eps) * torch.softmax(self.fc_rv(value), dim=-1) + self.domain_eps / self.n_domain
        v = torch.zeros(value.shape).to(self.device)
        # d = (dq + dk + dv)/3

        for i_d in range(self.n_domain):
            i_query = self.fc_q[i_d](query)  # [batch_size, query_len, hid_dim]
            i_key = self.fc_k[i_d](key)
            i_value = self.fc_v[i_d](value)

            i_dq = torch.repeat_interleave(dq[:, :, i_d], self.hid_dim, dim=1).view(i_query.shape)
            q += (i_query * i_dq)

            i_dk = torch.repeat_interleave(dk[:, :, i_d], self.hid_dim, dim=1).view(i_key.shape)
            k += (i_key * i_dk)

            i_dv = torch.repeat_interleave(dv[:, :, i_d], self.hid_dim, dim=1).view(i_value.shape)
            v += (i_value * i_dv)

        q /= self.n_domain
        k /= self.n_domain
        v /= self.n_domain

        q = q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # q = [batch_size, n_heads, query_len, hid_dim]
        # k = [batch_size, n_heads, key_len, hid_dim]
        # v = [batch_size, n_heads, value_len, hid_dim]

        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch_size, n_heads, query_len, key_len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), v)
        # x = [batch_size, n_heads, query_len, head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch_size, query_len, n_heads, head_dim]

        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch_size, query_len, hid_dim]

        x = self.fc_o(x)

        return x, attention, dq


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 n_domain,
                 domain_eps,
                 device,
                 max_length=100
                 ):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim).to(device)
        self.pos_embedding = nn.Embedding(max_length, hid_dim).to(device)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, n_domain, domain_eps, device) for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch_size, trg_len] output sequence
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        # trg = [batch_size, trg_len, hid_dim]

        for i, layer in enumerate(self.layers):
            if i == 0:
                trg, attention, domain = layer(trg, enc_src, trg_mask, src_mask)
            else:
                trg, attention, d = layer(trg, enc_src, trg_mask, src_mask)
                domain += d
        output = self.fc_out(trg)
        # output = [batch_size, trg_len, output_dim]

        return output, attention, domain/len(self.layers)


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 n_domain,
                 domain_eps,
                 device
                 ):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, n_domain, domain_eps, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, n_domain, domain_eps, device)
        self.position_wise_feedforward = PositionWiseFeedforwardLayer(hid_dim, pf_dim, dropout, n_domain, domain_eps, device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch_size, trg_len, hid_dim]
        _trg, _, att_domain = self.self_attention(trg, trg, trg, trg_mask)

        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch_size, trg_len, hid_dim]

        _trg, attention, enc_domain = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        _trg, ff_domain = self.position_wise_feedforward(trg)

        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg = [batch_size, trg_len, hid_dim]
        return trg, attention, (att_domain + enc_domain + ff_domain)/3


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 use_encoder_only,
                 device,
                 ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.use_encoder_only = use_encoder_only

    def make_src_mask(self, src):
        # src = [batch_size, src_len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch_size, trg_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_sub_mask = [trg_len, trg_len]

        trg_mask = trg_pad_mask * trg_sub_mask

        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src, enc_domain = self.encoder(src, src_mask)
        if self.use_encoder_only:
            output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
            return output, attention, enc_domain
        else:
            output, attention, dec_domain = self.decoder(trg, enc_src, trg_mask, src_mask)
            return output, attention, dec_domain
