import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixAttn(nn.Module):

  def __init__(self,linin,linout):
    super().__init__()
    self.attnlin = nn.Linear(linin,linout)

  def get_device(self):
    # return the device of the tensor, either "cpu" 
    # or number specifiing the index of gpu. 
    dev = next(self.parameters()).get_device()
    if dev == -1:
        return "cpu"
    return dev

  def forward(self,dec,emb):
    emb,elen = emb
    # dev = emb.get_device()
    # emask and emb should be in the same device 
    emask = torch.arange(0,emb.size(1)).unsqueeze(0).repeat(emb.size(0),1).long().to(self.get_device())
    
    emask = (emask >= elen.unsqueeze(1)).unsqueeze(1)
    decsmall = self.attnlin(dec)
    unnorm = torch.bmm(decsmall,emb.transpose(1,2))
    unnorm.masked_fill_(emask,-float('inf'))
    attn = F.softmax(unnorm,dim=2)
    out = torch.bmm(attn,emb)
    return out, attn

class BahdanauAttention(nn.Module):
    def __init__(self, num_units, query_size, memory_size):
        super(BahdanauAttention, self).__init__()

        self._num_units = num_units
        self._softmax = nn.Softmax()

        self.query_layer = nn.Linear(query_size, num_units, bias=False)
        self.memory_layer = nn.Linear(memory_size, num_units, bias=False)
        self.alignment_layer = nn.Linear(num_units, 1, bias=False)

    def _score(self, query, keys):
        # Put the query and the keys into Dense layer
        processed_query = self.query_layer(query)
        values = self.memory_layer(keys)

        # since the sizes of processed_query i [B x embedding],
        # we can't directly add it with the keys. therefore, we need
        # to add extra dimension, so the dimension of the query
        # now become [B x 1 x alignment unit size]
        extended_query = processed_query.unsqueeze(1)

        # The original formula is v * tanh(extended_query + values).
        # We can just use Dense layer and feed tanh as the input
        alignment = self.alignment_layer(F.tanh(extended_query + values))

        # Now the alignment size is [B x S x 1], We need to squeeze it
        # so that we can use Softmax later on. Converting to [B x S]
        return alignment.squeeze(2)

    def forward(self, query, keys):
        # Calculate the alignment score
        alignment_score = self._score(query, keys)

        # Put it into softmax to get the weight of every steps
        weight = F.softmax(alignment_score, dim=-1)

        # To get the context, this is the original formula
        # context = sum(weight * keys)
        # In order to multiply those two, we need to reshape the weight
        # from [B x S] into [B x S x 1] for broacasting.
        # The multiplication will result in [B x S x embedding]. Remember,
        # we want the score as the sum over all the steps. Therefore, we will
        # sum it over the 1st index
        context = weight.unsqueeze(2) * keys
        total_context = context.sum(1)

        return total_context, alignment_score


class LuongAttention(nn.Module):
    _SCORE_FN = {
        "dot": "_dot_score",
        "general": "_general_score",
        "concat": "_concat_score"
    }

    def __init__(self,
                 attention_window_size,
                 num_units,
                 query_size,
                 memory_size,
                 alignment="local",
                 score_fn="dot"):
        super(LuongAttention, self).__init__()

        if score_fn not in self._SCORE_FN.keys():
            raise ValueError()

        self._attention_window_size = attention_window_size
        self._softmax = nn.Softmax()
        self._score_fn = score_fn
        self._alignment = alignment

        self.query_layer = nn.Linear(query_size, num_units, bias=False)
        self.predictive_alignment_layer = nn.Linear(num_units, 1, bias=False)
        self.alignment_layer = nn.Linear(num_units, 1, bias=False)

        if score_fn == "general":
            self.general_memory_layer = nn.Linear(
                memory_size, query_size, bias=False)
        elif score_fn == "concat":
            self.concat_memory_layer1 = nn.Linear(
                2 * memory_size, num_units, bias=False)
            self.concat_memory_layer2 = nn.Linear(num_units, 1, bias=False)

    def _dot_score(self, query, keys):
        depth = query.size(-1)
        key_units = keys.size(-1)
        if depth != key_units:
            raise ValueError(
                "Incompatible inner dimensions between query and keys. "
                "Query has units: %d. Keys have units: %d. "
                "Dot score requires you to have same size between num_units in "
                "query and keys" % (depth, key_units))

        # Expand query to [B x 1 x embedding dim] for broadcasting
        extended_query = query.unsqueeze(1)

        # Transpose the keys so that we can multiply it
        tkeys = keys.transpose(1, 2)

        alignment = torch.matmul(extended_query, tkeys)

        # Result of the multiplication will be in size [B x 1 x embedding dim]
        # we can safely squeeze the dimension
        return alignment.squeeze(1)

    def _general_score(self, query, keys):
        weighted_keys = self.general_memory_layer(keys)
        extended_query = query.unsqueeze(1)
        weighted_keys = weighted_keys.transpose(1, 2)

        alignment = torch.matmul(extended_query, weighted_keys)
        return alignment.squeeze(1)

    def _concat_score(self, query, keys):
        expanded_query = query.unsqueeze(1).expand(*keys.size())
        concatenated_hidden = torch.cat([expanded_query, keys], dim=2)
        weighted_concatenated_hidden = self.concat_memory_layer1(
            concatenated_hidden)
        temp_score = F.tanh(weighted_concatenated_hidden)
        alignment = self.concat_memory_layer2(temp_score)

        return alignment.squeeze(2)

    def forward(self, query, keys, key_lengths):
        score_fn = getattr(self, self._SCORE_FN[self._score_fn])
        alignment_score = score_fn(query, keys)

        weight = F.softmax(alignment_score, dim=-1)

        if self._alignment == "local":
            extended_key_lengths = key_lengths.unsqueeze(1)
            preprocessed_query = self.query_layer(query)

            activated_query = F.tanh(preprocessed_query)
            sigmoid_query = F.sigmoid(
                self.predictive_alignment_layer(activated_query))
            predictive_alignment = extended_key_lengths * sigmoid_query

            ai_start = predictive_alignment - self._attention_window_size
            ai_end = predictive_alignment + self._attention_window_size

            std = torch.FloatTensor([self._attention_window_size / 2.]).pow(2)
            alignment_point_dist = (
                extended_key_lengths - predictive_alignment).pow(2)

            alignment_point_dist = (
                -(alignment_point_dist / (2 * std[0]))).exp()
            weight = weight * alignment_point_dist

            contexts = []
            for i in range(weight.size(0)):
                start = ai_start[i].int().data.numpy()[0]
                end = ai_end[i].int().data.numpy()[0]

                aligned_weight = weight[i, start:end]
                aligned_keys = keys[i, start:end]

                aligned_context = aligned_weight.unsqueeze(1) * aligned_keys
                contexts.append(aligned_context.sum(0))

            total_context = torch.stack(contexts, 0)
        elif self._alignment == "global":
            context = weight.unsqueeze(2) * keys
            total_context = context.sum(1)

        return total_context, alignment_score

    @property
    def attention_window_size(self):
        return self._attention_window_size


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 dropout_p=0.5,
                 h=8,
                 is_masked=False):
        super(MultiHeadAttention, self).__init__()

        if query_dim != key_dim:
            raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        self._key_dim = torch.tensor(key_dim,requires_grad=False).float()
        self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)
        self.bn = nn.BatchNorm1d(num_units)
        self.ln = nn.LayerNorm(num_units)

    def get_device(self):
        # return the device of the tensor, either "cpu" 
        # or number specifiing the index of gpu. 
        dev = next(self.parameters()).get_device()
        if dev == -1:
            return "cpu"
        return dev

    def forward(self, query, keys, mask=None):
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)

        # calculate QK^T
        attention = torch.matmul(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)

        # attention and _key_dim should be in the same device.
        attention = attention / torch.sqrt(self._key_dim).to(self.get_device())

        if mask is not None:
          mask = mask.repeat(self._h,1,1)
          attention.masked_fill_(mask,-float('inf'))
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        attention = F.dropout(attention, self._dropout_p)
        # multiplyt it with V
        attention = torch.matmul(attention, V)
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2)
        # residual connection
        attention += query
        # apply batch normalization
        #attention = self.bn(attention.transpose(1, 2)).transpose(1, 2)
        # apply layer normalization
        #attention = self.ln(attention)

        return attention
