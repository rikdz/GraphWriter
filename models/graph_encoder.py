import torch
import math
from torch import nn
from torch.nn import functional as F
from models.graphAttn import GAT
from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder
from models.attention import MultiHeadAttention


def gelu(x):
  return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Block(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.attn = MultiHeadAttention(args.hsz,args.hsz,args.hsz,h=4,dropout_p=args.drop)
    self.l1 = nn.Linear(args.hsz,args.hsz*4)
    self.l2 = nn.Linear(args.hsz*4,args.hsz)
    self.ln_1 = nn.LayerNorm(args.hsz)
    self.ln_2 = nn.LayerNorm(args.hsz)
    self.drop = nn.Dropout(args.drop)
    self.act = gelu

  def forward(self,q,k,m):
    q = self.attn(q,k,mask=m).squeeze(1)
    t = self.ln_1(q)
    q = self.drop(self.l2(self.act(self.l1(t))))
    q = self.ln_1(q+t)
    return q

class graph_encode(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.renc = nn.Embedding(args.rtoks,args.hsz)
    nn.init.xavier_normal_(self.renc.weight)
    #self.gat = StackedSelfAttentionEncoder(args.hsz,args.hsz,args.hsz,args.hsz,args.prop,args.heads,use_positional_encoding=False)

    self.gat = nn.ModuleList([MultiHeadAttention(args.hsz,args.hsz,args.hsz,h=4,dropout_p=args.drop) for _ in range(args.prop)])
    self.gat = nn.ModuleList([Block(args) for _ in range(args.prop)])
    self.prop = args.prop

  def pad(self,tensor,length):
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])

  def forward(self,adjs,rels,ents):
    vents,entlens = ents
    vents = torch.tensor(vents,requires_grad=False)
    vrels = [self.renc(x) for x in rels]
    gents = []
    grels = []
    glob = []
    for i,adj in enumerate(adjs):
      vgraph = torch.cat((vents[i][:entlens[i]],vrels[i]),0)
      N = vgraph.size(0)
      mask = (adj == 0).unsqueeze(1)
      for j in range(self.prop):
        ngraph = vgraph.repeat(N,1).view(N,N,-1)
        vgraph = self.gat[j](vgraph.unsqueeze(1),ngraph,mask)
        vgraph = F.layer_norm(vgraph,vgraph.size()[1:])
      gent = vgraph[:entlens[i],:]
      grel = vgraph[entlens[i]+1:,:]
      glob.append(vgraph[entlens[i]])
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
    glob = torch.stack(glob,0)
    return (gents,emask),glob,(grels,rmask)
