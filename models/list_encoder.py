import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from allennlp.modules.elmo import Elmo

class lseq_encode(nn.Module):

  def __init__(self,args,vocab=None,toks=None):
    super().__init__()
    if vocab:
      self.elmo = Elmo(args.options_file, args.weight_file, 1, dropout=0.5,vocab_to_cache=vocab)
      toks = len(vocab)
      sz = args.esz+512
      self.use_elmo = True
    else:
      self.use_elmo = False
      sz = args.esz
    self.lemb = nn.Embedding(toks,args.esz)
    nn.init.xavier_normal_(self.lemb.weight)
    self.input_drop = nn.Dropout(args.embdrop)
      
    self.encoder = nn.LSTM(sz,args.hsz//2,bidirectional=True,num_layers=args.layers,batch_first=True)

  def _cat_directions(self, h):
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h

  def forward(self,inp):
    l, ilens = inp
    learned_emb = self.lemb(l)
    learned_emb = self.input_drop(learned_emb)
    if self.use_elmo:
      elmo_emb = self.elmo(l,word_inputs=l)
      e = torch.cat((elmo_emb['elmo_representations'][0],learned_emb),2)
    else:
      e = learned_emb
    sent_lens, idxs = ilens.sort(descending=True)
    e = e.index_select(0,idxs)
    e = pack_padded_sequence(e,sent_lens,batch_first=True)
    e, (h,c) = self.encoder(e)
    e = pad_packed_sequence(e,batch_first=True)[0]
    e = torch.zeros_like(e).scatter(0,idxs.unsqueeze(1).unsqueeze(1).expand(-1,e.size(1),e.size(2)),e)
    h = h.transpose(0,1)
    h = torch.zeros_like(h).scatter(0,idxs.unsqueeze(1).unsqueeze(1).expand(-1,h.size(1),h.size(2)),h)
    return e,h

class list_encode(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.seqenc = lseq_encode(args,toks=args.vtoks)#,vocab=args.ent_vocab)
    #self.seqenc = lseq_encode(args,vocab=args.ent_vocab)

  def pad(self,tensor,length):
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])

  def forward(self,batch,pad=True):
    batch,phlens,batch_lens = batch
    batch_lens = tuple(batch_lens.tolist())
    _,enc = self.seqenc((batch,phlens))
    enc = enc[:,2:]
    enc = torch.cat([enc[:,i] for i in range(enc.size(1))],1)
    m = max(batch_lens)
    encs = [self.pad(x,m) for x in enc.split(batch_lens)]
    out = torch.stack(encs,0)
    return out

