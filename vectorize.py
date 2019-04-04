import torch
from collections import Counter
import dill
from torchtext import data
import pargs as arg
from copy import copy

class dataset:

  def __init__(self, args):
    args.path = args.datadir + args.data
    self.args = args
    '''
    if args.loadvocab:
      with open(args.datadir+"/"+args.loadvocab,'rb') as f:
        print("Loading Vocabs")
        self.fields = dill.load(f)
      self.INP,self.ENT,self.REL,self.TGT,self.OUTP = [x[1] for x in self.fields]
    else:
    '''
    self.mkVocabs(args)
    print("Vocab sizes:")
    for x in self.fields:
      try:
        print(x[0],len(x[1].vocab))
      except:
        pass
        
    '''
      if args.savevocab:
        with open(args.datadir+"/"+args.savevocab,'wb') as f:
          dill.dump(self.fields,f)
    '''

    if not args.eval:
      self.mkiters()

  def build_ent_vocab(self,path,unkat=0):
    ents = ""
    with open(path) as f:
      for l in f:
        ents +=  " "+l.split("\t")[1]
    itos = list(set(ents.split(" ")))
    itos[0] == "<unk>"; itos[1] == "<pad>"
    stoi = {x:i for i,x in enumerate(itos)}
    return itos,stoi

  def vec_ents(self,ex,field):
    # returns tensor and lens
    ex = [[field.stoi[x] if x in field.stoi else 0 for x in y.strip().split(" ")] for y in ex.split(";")]
    return self.pad_list(ex,1)
  
  def mkGraphs(self,r,ent):
    #convert triples to entlist with adj and rel matrices
    pieces = r.strip().split(';')
    x = [[int(y) for y in z.strip().split()] for z in pieces]
    rel = [2]
    #global root node
    adjsize = ent+1+(2*len(x))
    adj = torch.zeros(adjsize,adjsize)
    for i in range(ent):
      #adj[i,0]=1
      adj[ent+1,i]=1
    for i in range(adjsize):
      adj[i,i]=1
    for y in x:
      rel.extend([y[1]+3,y[1]+3+self.REL.size])
      a = y[0]
      b = y[2]
      c = ent+len(rel)-2
      d = ent+len(rel)-1
      adj[a,c] = 1 
      adj[c,b] = 1
      adj[b,d] = 1 
      adj[d,a] = 1
    rel = torch.LongTensor(rel)
    return (adj,rel)

  def mkVocabs(self,args):
    args.path = args.datadir + args.data
    self.INP = data.Field(sequential=True, batch_first=True,init_token="<start>", eos_token="<eos>",include_lengths=True)
    self.OUTP = data.Field(sequential=True, batch_first=True,init_token="<start>", eos_token="<eos>",include_lengths=True)
    self.TGT = data.Field(sequential=True, batch_first=True,init_token="<start>", eos_token="<eos>")
    self.ENT = data.RawField()
    self.REL = data.RawField()
    self.REL.is_target = False 
    self.ENT.is_target = False 
    self.fields=[("src",self.INP),("ent",self.ENT),("rel",self.REL),("tgt",self.TGT),("out",self.OUTP)]
    train = data.TabularDataset(path=args.path, format='tsv',fields=self.fields)

    print('building vocab')
    self.OUTP.build_vocab(train, min_freq=args.outunk)   
    self.TGT.vocab = copy(self.OUTP.vocab)
    specials = zip("method material otherscientificterm metric task".split(" "),range(40))
    for x in specials:
      s = "<"+x[0]+"_"+str(x[1])+">"
      self.TGT.vocab.stoi[s] = len(self.TGT.vocab.itos)+x[1]

    self.INP.build_vocab(train, min_freq=args.outunk)   
    '''
    self.INP.vocab.stoi['<pad>']=0
    self.INP.vocab.stoi['<unk>']=1
    self.INP.vocab.itos = ['<pad>','<unk>']+self.INP.vocab.itos[2:]
    '''

    self.REL.special = ['<pad>','<unk>','ROOT']
    with open(args.datadir+"/"+args.relvocab) as f:
      rvocab = [x.strip() for x in f.readlines()]
      self.REL.size = len(rvocab)
      rvocab += [x+"_inv" for x in rvocab]
      relvocab = self.REL.special + rvocab
    self.REL.itos = relvocab

    self.ENT.itos,self.ENT.stoi = self.build_ent_vocab(args.path)

    print('done')

  def listTo(self,l):
    return [x.to(self.args.device) for x in l]

  def fixBatch(self,b):
    #rel,rlens = self.relToBatch(rel)
    #adj,alens = self.adjToBatch(adj)
    ent,phlens = zip(*b.ent)
    ent,elens = self.adjToBatch(ent)
    ent = ent.to(self.args.device)
    #b.adj = (adj,alens)
    #b.rel = (rel,rlens)
    adj,rel = zip(*b.rel)
    b.rel = [self.listTo(adj),self.listTo(rel)]
    phlens = torch.cat(phlens,0).to(self.args.device)
    elens = elens.to(self.args.device)
    b.ent = (ent,phlens,elens)

    return b

  '''
  def entToBatch(self,inp):
    lens = torch.tensor(lens)
    m = torch.max(lens).item()
    data = [self.pad(x.transpose(0,1),m).transpose(0,1) for x in data]
    data = torch.cat(data,0)
    return data,lens

  def relToBatch(self,rel):
    lens = [x.size(0) for x in rel]
    m = max(lens)
    data = [self.pad(x,m).unsqueeze(0) for x in rel]
    data = torch.cat(data,0)
    return data,lens
  '''

  def adjToBatch(self,adj):
    lens = [x.size(0) for x in adj]
    m = max([x.size(1) for x in adj])
    data = [self.pad(x.transpose(0,1),m).transpose(0,1) for x in adj]
    data = torch.cat(data,0)
    return data,torch.LongTensor(lens)

  def bszFn(self,e,l,c):
    return c+len(e.out)

  def mkiters(self):
    args = self.args
    train = data.TabularDataset(path=args.path, format='tsv',fields=self.fields)
    self.trainsize = len(train.examples)
    valid = data.TabularDataset(path=args.path.replace("train","val"), format='tsv',fields=self.fields)
    for ds in [train,valid]:
      for x in ds:
        x.ent = self.vec_ents(x.ent,self.ENT)
        x.rel = self.mkGraphs(x.rel,len(x.ent[1]))
        '''
        try:
          assert(len(x.out)==len(x.tgt))
        except:
          print(len(x.out),len(x.tgt))
          print(x.out,x.tgt)
          mix = min([i for i in range(len(x.out)) if x.out[i] != x.tgt[i] and x.tgt[i][0]!="<"])
          print(mix,x.out[mix])
          z = zip(x.out,x.tgt)
          for k in z: 
            print(k)
          print(x.ent)
          exit()
        print('ok');exit()
        '''

    self.train_iter = data.Iterator(train,args.bsz*100,device=args.device,batch_size_fn=self.bszFn,sort_key=lambda x:len(x.out),repeat=False,train=True)
    self.val_iter= data.Iterator(valid,args.bsz*100,device=args.device,batch_size_fn = self.bszFn,sort_key=lambda x:len(x.out),sort=False,repeat=False,train=False)

  def mktestset(self, args):
    path = args.path.replace("train",'test')
    fields=self.fields
    dat = data.TabularDataset(path=path, format='tsv',fields=fields)
    dat.fields["rawent"] = data.RawField()
    for x in dat:
      x.rawent = x.ent.split(" ; ")
      x.rawent = (x.rawent, [[self.OUTP.vocab.stoi[y] if y in self.OUTP.vocab.stoi else 0 for y in z.split(" ")] for z in x.rawent])
      x.ent = self.vec_ents(x.ent,self.ENT)
      x.rel = self.mkGraphs(x.rel,len(x.ent[1]))
    dat_iter = data.Iterator(dat,args.vbsz,device=args.device,sort_key=lambda x:len(x.src), train=False, sort=False)
    return dat_iter

  def rev_ents(self,batch):
    vocab = self.NERD.vocab
    es = []
    for e in batch:
      s = [vocab.itos[y].split(">")[0]+"_"+str(i)+">" for i,y in enumerate(e) if vocab.itos[y] not in ['<pad>','<eos>']]
      es.append(s)
    return es

  def reverse(self,preds,ents,inp=False):
    ents = ents[0][0]
    vocab = self.TGT.vocab
    ss = []
    for i,x in enumerate(preds):
      s = ' '.join([vocab.itos[y] if y<len(vocab.itos) else "COPY" for j,y in enumerate(x)])   
      #s = ' '.join([vocab.itos[y] if y<len(vocab.itos) else ents[y-len(vocab.itos)] for j,y in enumerate(x)])   
      if "<eos>" in s: s = s.split("<eos>")[0]
      elif s[-1] not in ".?!":
        lastidx = [i for i,x in enumerate(s) if x in ".?!"]
        if lastidx:
          lastidx = max(lastidx)
          s = s[:lastidx+1]
      ss.append(s)
    if len(ss[0].split(' ')) == 0:
        return ['.']
    return ss

  def relfix(self,relstrs):
    mat = []
    for x in relstrs:
      pieces = x.strip().split(';')
      x = [[int(y)+len(self.REL.special) for y in z.strip().split()] for z in pieces]
      mat.append(torch.LongTensor(x).cuda())
    lens = [x.size(0) for x in mat]
    m = max(lens)
    mat = [self.pad(x,m) for x in mat]
    mat = torch.stack(mat,0)
    lens = torch.LongTensor(lens).cuda()
    return mat,lens

  def getEnts(self,entseq):
    newents = []
    lens = []
    for i,l in enumerate(entseq):
      l = l.tolist()
      if self.enteos in l:
        l = l[:l.index(self.enteos)]
      tmp = []
      while self.entspl in l:
        tmp.append(l[:l.index(self.entspl)])
        l = l[l.index(self.entspl)+1:]
      if l:
        tmp.append(l)
      lens.append(len(tmp))
      tmplen = [len(x) for x in tmp]
      m = max(tmplen)
      tmp = [x +([1]*(m-len(x))) for x in tmp]
      newents.append((torch.LongTensor(tmp).cuda(),torch.LongTensor(tmplen).cuda()))
    return newents,torch.LongTensor(lens).cuda()

  def listToBatch(self,inp):
    data, lens = zip(*inp)
    print(lens);exit()
    lens = torch.tensor(lens)
    m = torch.max(lens).item()
    data = [self.pad(x.transpose(0,1),m).transpose(0,1) for x in data]
    data = torch.cat(data,0)
    return data,lens


    

  def rev_rel(self,ebatch,rbatch):
    vocab = self.ENT.vocab
    for i,ents in enumerate(ebatch):
      es = []
      for e in ents:
        s = ' '.join([vocab.itos[y] for y in e])   
        es.append(s)
      rels = rbatch[i]
      for a,r,b in rels:  
        print(es[a],self.REL.itos[r],es[b])
      print()

  def pad_list(self,l,ent=1):  
    lens = [len(x) for x in l]
    m = max(lens)
    return torch.stack([self.pad(torch.tensor(x),m,ent) for x in l],0), torch.LongTensor(lens)

  def pad(self,tensor, length,ent=1):
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(ent)])

  def seqentmat(self,entseq):
    newents = []
    lens = []
    sms = []
    for l in entseq:  
      l = l.tolist()
      if self.enteos in l:
        l = l[:l.index(self.enteos)]
      tmp = []
      while self.entspl in l:
        tmp.append(l[:l.index(self.entspl)])
        l = l[l.index(self.entspl)+1:]
      if l:
        tmp.append(l)
      lens.append(len(tmp))
      m = max([len(x) for x in tmp])
      sms.append(m)
      tmp = [x +([0]*(m-len(x))) for x in tmp]
      newents.append(tmp)
    sm = max(lens)
    pm = max(sms)
    for i in range(len(newents)):
      tmp = torch.LongTensor(newents[i]).transpose(0,1)
      tmp = self.pad(tmp,pm,ent=0)
      tmp = tmp.transpose(0,1)
      tmp = self.pad(tmp,sm,ent=0)
      newents[i] = tmp
    newents = torch.stack(newents,0).cuda()
    lens = torch.LongTensor(lens).cuda()
    return newents,lens

if __name__=="__main__":
  args = arg.pargs()  
  ds = dataset(args)
  ds.getBatch()
  
