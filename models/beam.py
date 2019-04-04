import torch
from torch import nn
from torch.nn import functional as F

tt = torch.cuda if torch.cuda.is_available() else torch

class beam_obj():
  def __init__(self,initword,initscore,h,c,last):
    self.words = [initword]
    self.score = initscore
    self.h = h
    self.c = c
    self.last = last
    self.firstwords = [initword]
    self.prevent = 0
    self.isstart = False

class Beam():

  def __init__(self,words,scores,hs,cs,last,beamsz,k,vsz):
    self.beamsz = beamsz
    self.k = k
    self.beam = []
    for i in range(beamsz):
      self.beam.append(beam_obj(words[i].item(),scores[i].item(),hs[i],cs[i],last[i]))
    self.done = []
    self.vocabsz = vsz

  def sort(self,norm=True):
    if len(self.done)<self.beamsz:
      self.done.extend(self.beam)
      self.done = self.done[:self.beamsz]
    if norm:
      self.done = sorted(self.done,key=lambda x:x.score/len(x.words),reverse=True)
    else:
      self.done = sorted(self.done,key=lambda x:x.score,reverse=True)

  def dup_obj(self,obj):
    new_obj = beam_obj(None,None,None,None,None)
    new_obj.words = [x for x in obj.words]
    new_obj.score = obj.score
    new_obj.prevent = obj.prevent
    new_obj.firstwords = [x for x in obj.firstwords]
    new_obj.isstart = obj.isstart
    return new_obj

  def getwords(self):
    return tt.LongTensor([[x.words[-1]] for x in self.beam])

  def geth(self):
    return torch.cat([x.h for x in self.beam],dim=0)

  def getc(self):
    return torch.cat([x.c for x in self.beam],dim=0)

  def getlast(self):
    return torch.cat([x.last for x in self.beam],dim=0)

  def getscores(self):
    return tt.FloatTensor([[x.score] for x in self.beam]).repeat(1,self.k)

  def getPrevEnt(self):
    return [x.prevent for x in self.beam]

  def getIsStart(self):
    return [(i, self.beam[i].firstwords) for i in range(len(self.beam)) if self.beam[i].isstart]

  def update(self,scores,words,hs,cs,lasts):
    beam = self.beam
    scores = scores.squeeze()
    words = words.squeeze()
    k = self.k
    gotscores = self.getscores()
    scores = scores + self.getscores()
    scores, idx = scores.view(-1).topk(len(self.beam))
    newbeam = []
    for i,x in enumerate(idx):
      x = x.item()
      r = x//k; c = x%k
      w = words.view(-1)[x].item()
      new_obj = self.dup_obj(beam[r])
      if w == self.endtok:
        new_obj.score = scores[i]
        self.done.append(new_obj)
      else:
        if new_obj.isstart:
          new_obj.isstart = False
          new_obj.firstwords.append(w)
        if w >= self.vocabsz:
          new_obj.prevent = w
        if w == self.eostok:
          new_obj.isstart = True  
        new_obj.words.append(w)
        new_obj.score = scores[i]
        new_obj.h = hs[r,:].unsqueeze(0)
        new_obj.c = cs[r,:].unsqueeze(0)
        new_obj.last = lasts[r,:].unsqueeze(0)
        newbeam.append(new_obj)
    self.beam = newbeam
    return newbeam != []

