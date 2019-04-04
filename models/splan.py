import torch
from torch import nn
from torch.nn import functional as F
from models.attention import MultiHeadAttention

class splanner(nn.Module):
  def __init__(self,args):
    super().__init__()
    asz = 50
    self.emb = nn.Parameter(torch.zeros(1,3,asz))
    nn.init.xavier_normal_(self.emb)
    self.gru = nn.GRUCell(asz,asz)
    self.clin = nn.Linear(args.hsz,asz)
    self.klin = nn.Linear(args.hsz,asz)

  def attend(self,dec,emb,emask):
    dec = dec.unsqueeze(1)
    unnorm = torch.bmm(dec,emb.transpose(1,2))
    unnorm.masked_fill_(emask,-float('inf'))
    attn = F.softmax(unnorm,dim=2)
    return attn

  def plan_decode(self,hx,keys,mask,entlens):
    entlens = entlens[0]
    e = self.emb
    hx = self.clin(hx)
    keys = self.klin(keys)
    keysleft = keys.size(1)
    print(keysleft)
    keys = torch.cat((e,keys),1) 
    unmask = torch.zeros(hx.size(0),1,3).byte().cuda()
    print(mask)
    mask = torch.cat((unmask,mask),2)
    print(mask)
    ops = []
    prev = keys[:,1,:]
    while keysleft>1:
      hx = self.gru(prev,hx)
      print(hx.size(),keys.size())
      a = self.attend(hx,keys,mask)
      print(a)
      sel = a.max(2)[1].squeeze()
      print(sel)
      ops.append(keys[:,sel])
      if sel > 2 and sel != entlens:
        mask[0,0,sel]=1
        keysleft-=1
      if sel <= 2:
        mask[0,0,sel] = 1
      else:
        mask[0,0,:2] = 0
      if sel == entlens:
        mask[0,0,entlens] = 1
      else:
        mask[0,0,entlens] = 0

      prev = keys[:,sel]
    ops = torch.cat(ops,1)
    exit()
    return ops

  def forward(self,hx,keys,mask,entlens,gold=None):
    e = self.emb.repeat(hx.size(0),1,1)
    hx = self.clin(hx)
    keys = self.klin(keys)
    gold = gold[0]
    keys = torch.cat((e,keys),1) 
    gscaler = torch.arange(hx.size(0)).long().cuda()*keys.size(1)
    unmask = torch.zeros(hx.size(0),1,3).byte().cuda()
    mask = torch.cat((unmask,mask),2)
    ops = []
    goldup = gold.masked_fill(gold<3,0)
    for i,j in enumerate(entlens):
      goldup[i,j]=0
    prev = keys[:,1,:]
    for i in range(gold.size(1)):
      hx = self.gru(prev,hx)
      a = self.attend(hx,keys,mask)
      mask = mask.view(-1).index_fill(0,goldup[:,i]+gscaler,1).view_as(mask)
      ops.append(a)
      prev = keys.view(-1,keys.size(2))[gscaler]
    ops = torch.cat(ops,1)
    return ops

