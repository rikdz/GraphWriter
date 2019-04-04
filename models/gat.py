import torch
import numpy as np
from torch import nn
from beam import Beam
import models.encoders as encoders
from models.attn import attn
from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder

class model(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.encoder = encoders.gat_encode(args)
    self.decoder = decode(args)
    self.args = args

  def forward(self,outp,graph):
    ents,rels = self.encoder(graph)
    #rels = None
    h = torch.zeros(1,ents[1].size(0),self.args.hsz).cuda()
    c = torch.zeros(1,ents[1].size(0),self.args.hsz).cuda()
    decoded = self.decoder(outp,h,c,None,ents,rels)
    return decoded

  def beam_generate(self,title,entities,graph,nerd,beamsz=4,k=4):
    h,c,tembs,vembs,gembs = self.encoder(title,entities,graph)
    beam = self.decoder.beam_generate(h,c,tembs,vembs,gembs,nerd,beamsz,k)
    return beam


class decode(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.args = args
    #attns
    self.t_attn = attn(args.hsz,args.hsz)
    self.e_attn = attn(args.hsz,args.hsz)
    self.g_attn = attn(args.hsz,args.hsz)
    self.h_attn = attn(args.hsz,args.hsz)
    # decoder
    self.Embedding = nn.Embedding(args.ntoks,args.hsz)
    #nn.init.xavier_normal_(self.Embedding.weight)
    self.dlstm = nn.LSTM(args.hsz*2,args.hsz,batch_first=True)
    self.outlin = nn.Linear(args.hsz*2,args.ntoks)
    self.attnlin = nn.Linear(args.hsz,args.hsz)
    self.switch = nn.Linear(args.hsz,1)
    self.ln = nn.LayerNorm(args.hsz)

    self.maxlen = 200
    self.starttok = args.starttok
    self.endtok = None

  def hierattn(self,dec,embs):
    embs = [x for x in embs if x]
    contexts = []
    weights = []
    for attn,emb in embs:
      context, weight = attn(dec,emb)
      contexts.append(context)
      weights.append(weight)
    if len(contexts)>=1:
      contexts = torch.cat(contexts,1)
      combined, hweight = self.h_attn(dec,(contexts,torch.ones(context.size(0),contexts.size(1)).cuda()))
    else:
      combined,hweight = contexts[0],weights[0]
    return combined, weights[-1], hweight
    
  def forward(self,outp,h,c,tembs,vembs,gembs):
    embs = [x for x in [(self.t_attn,tembs),(self.g_attn,gembs),(self.e_attn,vembs)] if x[1] is not None]
    outp = self.Embedding(outp[0])
    last = h.transpose(0,1)
    outs = []
    vweights = []
    scalars = []
    zero_vec = 1e-6*torch.ones(outp.size(0),self.args.ntoks+vembs[1].size(1)).cuda()
    for i in range(outp.size(1)):
      enc = outp[:,i,:].unsqueeze(1)
      decin = torch.cat((enc,last),2)
      decout,(h,c) = self.dlstm(decin,(h,c))
      decout = self.ln(decout)
      last, vweight, hweight = self.hierattn(decout,embs)
      out = torch.cat((decout,last),2).squeeze(1)
      vweight = vweight.squeeze(1)
      decoded = self.outlin(out)
      scalar = torch.sigmoid(self.switch(h.squeeze()))
      decoded = torch.softmax(decoded,1)
      decoded = torch.mul(decoded,1-scalar.expand_as(decoded))
      vweight = torch.mul(vweight,scalar.expand_as(vweight))
      decoded = torch.cat([decoded,vweight],1)
      decoded += zero_vec
      outs.append(decoded)
    decoded = torch.stack(outs,1)
    return decoded.log()

  def emb_w_vertex(self,outp,vertex):
    mask = outp>=self.args.ntoks
    if mask.sum()>0:
      idxs = (outp-self.args.ntoks)
      idxs = idxs[mask]
      verts = vertex.index_select(1,idxs)
      outp.masked_scatter_(mask,verts)

    return outp

  def beam_generate(self,h,c,tembs,vembs,gembs,nerd,beamsz,k):
    #h,c,tembs,vembs,gembs,rembs = self.encode_inputs(title,entities,graph)
    #h,c,tembs,vembs,gembs = self.encode_inputs(title,entities,graph)
    embs = [x for x in [(self.t_attn,tembs),(self.g_attn,gembs),(self.e_attn,vembs)] if x[1] is not None]

    outp = torch.LongTensor(vembs[0].size(0),1).fill_(self.starttok).cuda()
    last = h.transpose(0,1)
    outputs = []
    beam = None
    for i in range(self.maxlen):
      outp = self.emb_w_vertex(outp.clone(),nerd)
      enc = self.Embedding(outp)
      decin = torch.cat((enc,last),2)
      decout,(h,c) = self.dlstm(decin,(h,c))
      last, vweight, _ = self.hierattn(decout,embs)
      scalar = torch.sigmoid(self.switch(h))
      outs = torch.cat((decout,last),2)
      decoded = self.outlin(outs.contiguous().view(-1, self.args.hsz*2))
      decoded = decoded.view(outs.size(0), outs.size(1), self.args.ntoks)
      decoded = torch.softmax(decoded,2)
      decoded[:,:,0].fill_(0)
      decoded[:,:,1].fill_(0)
      scalars = scalar.transpose(0,1)
      decoded = torch.mul(decoded,1-scalars.expand_as(decoded))
      vweights = torch.mul(vweight,scalars.expand_as(vweight))
      decoded = torch.cat([decoded,vweights],2)

      zero_vec = 1e-6*torch.ones_like(decoded)
      decoded += zero_vec
      decoded = decoded.log()
      scores, words = decoded.topk(dim=2,k=k)
      #scores = scores.transpose(0,1); words = words.transpose(0,1)
      if not beam:
        beam = Beam(words.squeeze(),scores.squeeze(),[h for i in range(beamsz)],
                  [c for i in range(beamsz)],[last for i in range(beamsz)],beamsz,k)
        beam.endtok = self.endtok
        newembs = []
        for a,x in embs:
          tmp = (x[0].repeat(len(beam.beam),1,1),x[1].repeat(len(beam.beam),1))
          newembs.append((a,tmp))
        embs = newembs
      else:
        if not beam.update(scores,words,h,c,last):
          break
        newembs = []
        for a,x in embs:
          tmp = (x[0][:len(beam.beam),:,:],x[1][:len(beam.beam)])
          newembs.append((a,tmp))
        embs = newembs
      outp = beam.getwords()
      h = beam.geth()
      c = beam.getc()
      last = beam.getlast()

    return beam

