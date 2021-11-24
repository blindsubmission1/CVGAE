import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os, sys

import dataset

from train import *
from evaluator import *
from graph_model import *
from retrieval_model import *
from utils import *

from flags import args

torch.manual_seed(12345)

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
  
DATA_ROOT_DIR = './data'
GCZSL_DS_ROOT = {
  'MIT': DATA_ROOT_DIR+'/mit-states-natural',
  'UT':  DATA_ROOT_DIR+'/ut-zap50k-natural',
  'CGQA': DATA_ROOT_DIR+'/cgqa-natural'}


model_name = args.name

dataset_name = args.ds.upper()
open_world = False
if not args.ir_model: 
  open_world = not args.close_world
  print ("this is the world we are living in: "+ str (open_world))
train_only = args.train_only
cpu_eval = args.cpu_eval

if os.path.isfile(os.path.join(GCZSL_DS_ROOT[dataset_name], 'resnet'+args.resnet+'.pt')):
  feat_file = 'resnet'+args.resnet+'.pt'
  resnet_name = None
  print ("Image features are loaded. The same fixed image feature file as used by CompCos, CGE and other baselines for fair comparison.")
else:
  feat_file = None
  resnet_name = 'resnet'+args.resnet
with_image = resnet_name is not None
static_inp = not args.resnet_trainable
resnet_lr = args.resnet_lr

lr = args.lr
weight_decay = args.wd
num_epochs = args.epoch
batch_size = args.bs

eval_every = args.eval_every

hparam = HParam()
hparam.add_dict({'lr': lr, 'batchsize': batch_size, 'wd': weight_decay,
                'open_world': open_world, 'train_only': train_only})
if resnet_name and not static_inp:
  hparam.add_dict({'resnet': resnet_name, 'resnet_lr': resnet_lr})

# =======   Dataset & Evaluator  =======
if args.ir_model:
  rand_sampling = True
  if dataset_name=='MIT':
    ignore_objs = [
                'armor', 'bracelet', 'bush', 'camera', 'candy', 'castle',
                'ceramic', 'cheese', 'clock', 'clothes', 'coffee', 'fan', 'fig',
                'fish', 'foam', 'forest', 'fruit', 'furniture', 'garden', 'gate',
                'glass', 'horse', 'island', 'laptop', 'lead', 'lightning',
                'mirror', 'orange', 'paint', 'persimmon', 'plastic', 'plate',
                'potato', 'road', 'rubber', 'sand', 'shell', 'sky', 'smoke',
                'steel', 'stream', 'table', 'tea', 'tomato', 'vacuum', 'wax',
                'wheel', 'window', 'wool'
                ]
  elif dataset_name=='UT':
    ignore_objs = [
                'Shoes.Boat.Shoes',
                'Boots.Knee.High'
                ]
  else:
    ignore_objs = []
else:
  rand_sampling=False
  ignore_objs = []

train_dataloader = dataset.get_dataloader(dataset_name, 'train', feature_file=feat_file, batchsize=batch_size, open_world=open_world, 
                                          train_only=train_only, shuffle=True, random_sampling=rand_sampling, ignore_objs=ignore_objs)
val_set = 'test' 
val_dataloader = dataset.get_dataloader(dataset_name, val_set, feature_file=feat_file, batchsize=batch_size,
                                        open_world=open_world, random_sampling=rand_sampling, ignore_objs=ignore_objs)
dset = train_dataloader.dataset
nbias = 20


# ======  Load HParam from checkpoint =======

log_path = os.path.join('logs/', dataset_name, model_name)
model_path = os.path.join(log_path, model_name+'.pt')

if model_name == 'tmp' and os.path.isfile(model_path):
  os.remove(model_path)
try:
  checkpoint = torch.load(model_path)
except FileNotFoundError:
  checkpoint = None

if checkpoint and 'hparam_dict' in checkpoint:
  hparam.add_dict(checkpoint['hparam_dict'])
  hparam.freeze()

# ====     Model & Loss    ========
# Same initial graph embeddings as CGE for fairness reasons
graph_path = os.path.join(dset.root, 'graph_primitive.t7') 


if args.ir_model:
  criterion = cvgae_loss_IR 
  val_criterion = dummy_loss
  model = CVGAEIR(hparam, dset, graph_path=graph_path, train_only=train_only, resnet_name=resnet_name, static_inp=static_inp).to(dev)
  val_evaluator = IREvaluator(cpu_eval)
  target_metric = 'IR_Rec/top1'
else:

  model = CVGAE(hparam, dset, graph_path=graph_path, train_only=train_only, resnet_name=resnet_name).to(dev)
  criterion = cvgae_loss
  val_criterion = criterion
  val_evaluator = Evaluator(val_dataloader, nbias, cpu_eval, open_world = open_world)
  target_metric = 'AUC'
# hparam.add_dict(criterion.hparam_dict)


if not static_inp:
  model_params, resnet_params = [], []
  for name, param in model.named_parameters():
    if name.split('.')[0] == 'resnet':
      resnet_params.append(param)
    else:
      model_params.append(param)
  params = [{'params': model_params}]  
  params.append({'params': resnet_params, 'lr': resnet_lr})
  optimizer = torch.optim.Adam(params, lr=hparam.lr)
else:
  optimizer = torch.optim.Adam(model.parameters(), lr=hparam.lr)


# === Restore model and logger from Checkpoint ===
curr_epoch = 0
best = {target_metric:-1, 'best_epoch':-1}

if checkpoint:
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  if target_metric in checkpoint:
    best[target_metric] = checkpoint[target_metric]
  if 'epoch' in checkpoint:
    curr_epoch = checkpoint['epoch']
  del checkpoint
  print('Model loaded.')

if model_name == 'tmp':
  logger = DummyLogger()
else:
  logger = SummaryWriter(log_path)
print(f"Logging to: {logger.log_dir}")


# =====   Evaluation  ======
if args.eval:
  summary, _ = evaluate(model, val_criterion, val_dataloader, val_evaluator, open_world, cpu_eval)
  for key, value in summary.items():
    print(f'{key}:{value:.4f}|', end='')
  print()
  sys.exit(0)

# ====     Train    ========
try:
  train(model_name, model, hparam, optimizer, criterion, val_criterion, num_epochs, batch_size, train_dataloader, val_dataloader, logger, val_evaluator, target_metric,
        curr_epoch=curr_epoch, best=best, open_world=open_world, eval_every=eval_every, cpu_eval=cpu_eval)
except KeyboardInterrupt:
  print("Training stopped.")
finally:
  logger.add_text(f'hparam/{model_name}', repr(hparam.hparam_dict | best))
  logger.flush()
  logger.close()
  print(f'Best {target_metric}: {best[target_metric]}')