from . import GCZSL_dataset
from torch.utils.data import DataLoader
import numpy as np

DATA_ROOT_DIR = './data'

GCZSL_DS_ROOT = {
  'MIT': DATA_ROOT_DIR+'/mit-states-natural',
  'UT':  DATA_ROOT_DIR+'/ut-zap50k-natural',
  'CGQA': DATA_ROOT_DIR+'/cgqa-natural'}

def get_dataloader(dataset_name, phase, feature_file, batchsize, num_workers=0, open_world=True, train_only=False, 
                   random_sampling=False,ignore_attrs=[], ignore_objs=[], shuffle=None, **kwargs):
  
  dataset =  GCZSL_dataset.CompositionDatasetActivations(
        name = dataset_name,
        root = GCZSL_DS_ROOT[dataset_name], 
        phase = phase,
        feat_file = feature_file,
        open_world = open_world,
        train_only = train_only,
        random_sampling = random_sampling,
        ignore_attrs=ignore_attrs,
        ignore_objs=ignore_objs,
          **kwargs)

  if shuffle is None:
    shuffle = (phase=='train')

  return DataLoader(dataset, batchsize, shuffle, num_workers=num_workers)


    

