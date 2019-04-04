import torch
from torch import nn
from torch.nn import functional as F

class attn(nn.Module):

  def __init__(self,linin,linout):
    super(attn, self).__init__()
    self.attnlin = nn.Linear(linin,linout)

  def forward(self,dec,emb):
    emb,emask = emb #; elen = elen.cuda()
    emask = (emask == 0).unsqueeze(1)
    #emask = torch.arange(0,emb.size(1)).unsqueeze(0).repeat(emb.size(0),1).long().cuda()
    #emask = (emask >= elen.unsqueeze(1)).unsqueeze(1)
    decsmall = self.attnlin(dec)
    unnorm = torch.bmm(decsmall,emb.transpose(1,2))
    unnorm.masked_fill_(emask,-float('inf'))
    attn = F.softmax(unnorm,dim=2)
    out = torch.bmm(attn,emb)
    return out, attn
