import torch
import argparse

def dynArgs(args,ds):
  args.ntoks = len(ds.OUTP.vocab)
  args.tgttoks = len(ds.TGT.vocab)
  args.ninput = len(ds.INP.vocab)
  args.vtoks = len(ds.ENT.itos)
  args.rtoks = len(ds.REL.itos)
  args.starttok = ds.OUTP.vocab.stoi["<start>"]
  args.dottok = ds.OUTP.vocab.stoi["."]
  args.ent_vocab = ds.ENT.itos
  args.inp_vocab = ds.INP.vocab.itos
  args.lrchange = (args.lrhigh - args.lr)/args.lrstep
  args.esz = args.hsz
  return args

def pargs():
  parser = argparse.ArgumentParser(description='Graph Doc Plan')

  #model
  parser.add_argument("-model",default="graph",help="model types: 'graph' for graph transformer (default), 'gat' for graph attention network, 'ents' for entity-only model." )
  parser.add_argument("-esz",default=500,type=int,help='embedding size')
  parser.add_argument("-hsz",default=500,type=int,help="hidden state size")
  parser.add_argument("-prop",default=6,type=int,help="number of layers/blocks")
  parser.add_argument("-title",action='store_true',help="use title as additional input")
  parser.add_argument("-drop",default=0.1,type=float,help="dropout rate")
  parser.add_argument("-embdrop",default=0,type=float,help="embedding dropout")
  parser.add_argument("-layers",default=2,type=int,help='encoder lstm layers')
  #parser.add_argument("-blockdrop",default=0.1)
  #parser.add_argument("-gdrop",default=0.3,type=float)
  #parser.add_argument("-attnheads",default=3,type=int,
  #parser.add_argument("-elmoVocab",default="../data/elmoVocab.txt",type=str)
  #parser.add_argument("-elmo",action='store_true')
  #parser.add_argument("-heads",default=4,type=int)



  # training and loss
  parser.add_argument("-cl",default=None,type=float,help="Coverage loss")
  parser.add_argument("-bsz",default=32,type=int)
  parser.add_argument("-epochs",default=20,type=int)
  parser.add_argument("-clip",default=1,type=float,help='clip grads')
  parser.add_argument("-t1size",default=32,type=int,help="batch size for short targets")
  parser.add_argument("-t2size",default=16,type=int,help="batch size for medium length targets")
  parser.add_argument("-t3size",default=8,type=int,help="batch size for long targets")

  #optim
  '''
  parser.add_argument('-lr_warmup', type=float, default=0.002)
  parser.add_argument('-lr_schedule', type=str, default='warmup_linear')
  parser.add_argument("-lr",default=6.25e-5,type=float)
  parser.add_argument("-sgdlr",default=0.1,type=float)
  parser.add_argument('-b1', type=float, default=0.9)
  parser.add_argument('-b2', type=float, default=0.999)
  parser.add_argument('-e', type=float, default=1e-8)
  parser.add_argument('-l2', type=float, default=0.01)
  parser.add_argument('-vector_l2', action='store_true')
  '''
  parser.add_argument("-lr",default=0.1,type=float,help='learning rate')
  parser.add_argument("-lrhigh",default=0.5,type=float,help="high learning rate for cycling")
  parser.add_argument("-lrstep",default=4, type=int,help='steps in cycle')
  parser.add_argument("-lrwarm",action="store_true",help='use cycling learning rate')
  parser.add_argument("-lrdecay",action="store_true",help="use learning rate decay")


  #parser.add_argument('-max_grad_norm', type=int, default=1)

  #data
  parser.add_argument("-nosave",action='store_false',help='dont save')
  parser.add_argument("-save",required=True,help="where to save model")
  parser.add_argument("-outunk",default=5,type=int,help="unk @ for targets")
  parser.add_argument("-entunk",default=5,type=int,help="unk @ for entity vocabulary")
  parser.add_argument("-datadir",default="data/")
  parser.add_argument("-data",default="preprocessed.train.tsv",help="preprocessed data")
  parser.add_argument("-traindata",default="preprocessed.train.tsv",help="preprocessed train data")
  parser.add_argument("-relvocab",default="relations.vocab",type=str,help='vocabulary of graph relations')
  parser.add_argument("-savevocab",default=None,type=str)
  parser.add_argument("-loadvocab",default=None,type=str)

  #eval
  parser.add_argument("-eval",action='store_true')


  #inference
  parser.add_argument("-max",default=200,type=int,help="max length of generation")
  parser.add_argument("-test",action='store_true')
  #parser.add_argument("-sample",action='store_true')
  parser.add_argument("-inputs",default="../data/fullGraph.test.tsv",type=str)


  parser.add_argument("-sparse",action='store_true',help="sparse graphs (NOT CURRENTLY IMPLEMENTED)")
  parser.add_argument("-plan",action='store_true',help="plan and write (NOT IMPLEMENTED)")
  parser.add_argument("-ckpt",default=None,type=str,help='load checkpoint')
  parser.add_argument("-plweight",default=0.2,type=float,help="plan weight (NOT IMPLEMENTED)")
  parser.add_argument("-entdetach",action='store_true',help='dont backprop into entity embeddings')

  parser.add_argument("-gpu",default=0,type=int)
  args = parser.parse_args()
  if args.gpu == -1:
    args.gpu = 'cpu'
  args.device = torch.device(args.gpu)

  #args.options_file = "../elmo/elmo_2x2048_256_2048cnn_1xhighway_options.json"
  #args.weight_file = "../elmo/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"

  return args
