from typing import *
import os
import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

import dataset
from evaluator import _IREvaluator

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 
  

    
def tqdm_iter(curr_epoch, total_epoch, dataloader):
  postfix = f'Train: epoch {curr_epoch}/{total_epoch}'
  return tqdm.tqdm(enumerate(dataloader), 
                 total=len(dataloader), position=0, leave=True, postfix=postfix)


def log_summary(summary, logger, epoch):
  for key, value in summary.items():
    if 'Op' in key:
      logger.add_scalar(key[2:]+'/op', value, epoch)
    elif 'Cw' in key:
      logger.add_scalar(key[2:]+'/cw', value, epoch)
    else:
      logger.add_scalar('Acc/'+key, value, epoch)
      
      
def cw_output_converter(output, dataloader, cpu_eval=False):
  val_dev = 'cpu' if cpu_eval else dev
  dataset = dataloader.dataset
  if isinstance(output[0], tuple):
    output = list(zip(*output))[0]
  output = torch.cat(output).to(val_dev)
  batch_size = output.size(0)
  nattr, nobj = len(dataset.attrs), len(dataset.objs)
  new_output = torch.ones((batch_size, nattr*nobj)).to(val_dev) * 1e-10
  op_idx = [dataset.op_pair2idx[pair] for pair in dataset.pairs]
  new_output[:, op_idx] = output
  return new_output


def evaluate(net, val_criterion, val_dataloader, evaluator, open_world, cpu_eval):
  test_loss = defaultdict(lambda: 0)
  outputs = []
  attr_labels, obj_labels = [], []
  val_dev = 'cpu' if cpu_eval else dev
  net.eval()
  with torch.no_grad():
    for i, sample in tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
      output = net(sample)
      total_loss, loss_dict = val_criterion(output, sample)
      for key, loss in loss_dict.items():
        test_loss[key] += loss.item()
      if isinstance(output, tuple):
        output = tuple([x.to(val_dev) if isinstance(x, torch.Tensor) else x for x in output])
      else:
        output = output.to(val_dev)
      outputs.append(output)
      attr_labels.append(sample[1])
      obj_labels.append(sample[2])

  attr_labels = torch.cat(attr_labels).to(val_dev)
  obj_labels = torch.cat(obj_labels).to(val_dev)

  if not open_world and not isinstance(evaluator, _IREvaluator):
    outputs = cw_output_converter(outputs, val_dataloader, cpu_eval)

  if open_world:
    fscore = output[2].fscore
    summary = evaluator.eval_output(outputs, attr_labels, obj_labels, fscore)
  else:
    summary = evaluator.eval_output(outputs, attr_labels, obj_labels)
  return summary, test_loss


def train(model_name, net, hparam, optimizer, criterion, val_criterion, num_epochs, batch_size, train_dataloader, val_dataloader, logger,
          evaluator, target_metric, curr_epoch=0, best=None, open_world=True, eval_every=1, cpu_eval=False):

  iters = len(train_dataloader)
  if not best:
    best = defaultdict(lambda: -1)
  
  for epoch in range(curr_epoch, curr_epoch+num_epochs):
    
    # ==== Training ====
    running_loss = defaultdict(lambda : 0)
    
    net.train()
    for i, sample in tqdm_iter(epoch, curr_epoch+num_epochs, train_dataloader):
      optimizer.zero_grad()
      if len(sample[0]) == 1:
        # Batchnorm doesn't accept batch with size 1
        continue
      output = net(sample)
      total_loss, loss_dict = criterion(output, sample)
      total_loss.backward()
      optimizer.step()

      for key, loss in loss_dict.items():
        running_loss[key] += loss.item()
      if i % 100 == 99:
        for key, loss in running_loss.items():
          logger.add_scalar(f'{key}/train', loss/i, epoch*len(train_dataloader)//100+i//100)
    
    if epoch % eval_every != 0:
      continue
      
    # ==== Evaluation ====
    summary, test_loss = evaluate(net, val_criterion, val_dataloader, evaluator, open_world, cpu_eval)

    # ==== Logging ====
    log_summary(summary, logger, epoch)
    print("Train: ", end='')
    for key, loss in running_loss.items():
      print(f"{key}: {loss/len(train_dataloader)}", end=' - ')
    print()
    
    print("Test: ", end='')
    for key, loss in test_loss.items():
      loss /= len(val_dataloader) 
      logger.add_scalar(f'{key}/test', loss, epoch)
      print(f"{key}: {loss}", end=' - ')
    print()
    print("Performance on the test set: ", end='')
    print()
    for key, value in summary.items():
      print(f'{key}:{value:.4f}|', end='')
   
    print()
    if summary[target_metric] > best[target_metric]:
      best[target_metric] = summary[target_metric]
      best['best_epoch'] = epoch
      if logger.log_dir:
        torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    target_metric: summary[target_metric],
                    'epoch': epoch+1,
                    'model_str': repr(net),
                    'hparam_dict': hparam.hparam_dict
                    }, os.path.join(logger.log_dir, model_name+'.pt'))
        
  print("Finished training.")