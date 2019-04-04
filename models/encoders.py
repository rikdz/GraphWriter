import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from allennlp.modules.elmo import Elmo
from models.graphAttn import GAT
from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder

class encode_inputs(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.args = args
    if not args.notitle:
      self.seq_encode = seq_encode(args,args.inp_vocab)
    self.list_encode = list_encode(args)
    self.graph_encode = gat_encode(args)

    self.wavg = weighted_avg(args.hsz)
    
  def forward(self,title,entities=None,graph=None):
    tembs = None; vembs = None; gembs = None; 
    if not self.args.notitle:
      tembs,(h,c) = self.seq_encode(title)
    else:
      h = torch.zeros(1,entities[1].size(0),self.args.hsz).cuda()
      c = torch.zeros(1,entities[1].size(0),self.args.hsz).cuda()
    if entities:
      elens = entities[1] 
      vembs = self.list_encode(entities)
      emask = torch.arange(0,vembs.size(1)).unsqueeze(0).repeat(vembs.size(0),1).long().cuda()
      emask = (emask < elens.unsqueeze(1))
      vembs = (vembs, emask)
    if graph:
      gembs = self.graph_encode(graph) 
    return h,c,tembs,vembs,gembs

class seq_encode(nn.Module):

  def __init__(self,args,vocab):
    super().__init__()
    toks = len(vocab)
    #self.elmo = Elmo(args.options_file, args.weight_file, 1, dropout=0.5,vocab_to_cache=vocab)
    self.emb = nn.Embedding(toks,512)
    nn.init.xavier_normal_(self.emb.weight)
    self.input_drop = nn.Dropout(args.embdrop)
    self.encoder = nn.LSTM(512,args.hsz//2,bidirectional=True,num_layers=args.layers,batch_first=True)
    #self.encoder = nn.LSTM(1024,args.hsz//2,bidirectional=True,num_layers=args.layers,batch_first=True)

  def _cat_directions(self, h):
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h

  def forward(self,inp):
    l, ilens = inp
    #elmo_emb = self.elmo(l,word_inputs=l)
    learned_emb = self.emb(l)
    #e = torch.cat((elmo_emb['elmo_representations'][0],learned_emb),2)
    e = learned_emb
    e = self.input_drop(e)
    e = pack_padded_sequence(e,ilens,batch_first=True)
    e, (h,c) = self.encoder(e)
    h = self._cat_directions(h)
    c = self._cat_directions(c)
    e = pad_packed_sequence(e,batch_first=True)
    return e, (h,c)

class weighted_avg(nn.Module):
  def __init__(self,hsz):
    super().__init__()
    self.W = nn.Parameter(torch.zeros(size=(hsz, 1)))
    nn.init.xavier_uniform_(self.W, gain=1.414)

  def forward(self,enc,mask):
    weights = torch.bmm(enc,self.W.unsqueeze(0).repeat(enc.size(0),1,1)).squeeze(2)
    weights.masked_fill_(mask,float('-inf'))
    weights = torch.softmax(weights,1).unsqueeze(2)
    vecs = torch.bmm(enc.transpose(1,2),weights).squeeze(2)
    return vecs

class list_encode(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.seqenc = seq_encode(args,args.ent_vocab)
    self.head = weighted_avg(args.hsz)

  def pad(self,tensor,length):
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])

  def forward(self,batch,pad=True):
    batch,batch_lens = batch
    m = max(batch_lens)
    out = []
    for l in batch:
      enc,_ = self.seqenc(l)
      enc = enc[0]
      mask = torch.arange(0,enc.size(1)).unsqueeze(0).repeat(enc.size(0),1).long().cuda()
      mask = (mask <= l[1].unsqueeze(1))
      mask = mask==0
      v = self.head(enc,mask)
      if pad:
        v = self.pad(v,m)
      out.append(v)
    if pad:
      out = torch.stack(out,0)
    return out

class gat_encode(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.lenc = list_encode(args)
    self.renc = nn.Embedding(args.rtoks,args.hsz)
    nn.init.xavier_normal_(self.renc.weight)
    #self.gat = GAT(args.hsz, args.hsz, args.hsz, args.gdrop, 0.2, 3)
    self.gat = StackedSelfAttentionEncoder(args.hsz,512,512,args.hsz,args.prop,4,use_positional_encoding=False)
    self.prop = args.prop

  def pad(self,tensor,length):
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])

  def forward(self,graph):
    ents,adjs,rels = graph
    vents = self.lenc(ents,pad=False)
    vrels = [self.renc(x) for x in rels]
    gents = []
    grels = []
    for i,adj in enumerate(adjs):
      vgraph = torch.cat((vents[i],vrels[i]),0)
      N = vgraph.size(0)
      vgraph = vgraph.repeat(N,1).view(N,N,-1)
      vgraph = self.gat(vgraph,adj)
      vgraph = torch.stack([vgraph[i,i,:] for i in range(N)])
      gent = vgraph[:vents[i].size(0),:]
      grel = vgraph[vents[i].size(0):,:]
      gents.append(gent)
      grels.append(grel)
    elens = [x.size(0) for x in gents]
    gents = [self.pad(x,max(elens)) for x in gents]
    gents = torch.stack(gents,0)
    elens = torch.LongTensor(elens)
    emask = torch.arange(0,gents.size(1)).unsqueeze(0).repeat(gents.size(0),1).long()
    emask = (emask <= elens.unsqueeze(1)).cuda()
    rlens = [x.size(0) for x in grels]
    grels = [self.pad(x,max(rlens)) for x in grels]
    grels = torch.stack(grels,0)
    rlens = torch.LongTensor(rlens)
    rmask = torch.arange(0,grels.size(1)).unsqueeze(0).repeat(grels.size(0),1).long()
    rmask = (rmask <= rlens.unsqueeze(1)).cuda()
    return (gents,emask),(grels,rmask)
