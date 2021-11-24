'''Code adapted from Symnet: https://arxiv.org/abs/2004.00587'''

import numpy as np
import torch, torchvision
import os, logging, pickle, sys
import tqdm
from itertools import product
import os.path as osp

from . import data_utils


class CompositionDatasetActivations(torch.utils.data.Dataset):

    def __init__(self, name, root, phase, feat_file, split='compositional-split', with_image=False, transform_type='normal', 
                 open_world=True, train_only=False, random_sampling=False, ignore_attrs=[], ignore_objs=[]):
        self.name = name
        self.root = root
        self.phase = phase
        self.split = split
        self.with_image = feat_file is None
        self.random_sampling = random_sampling

        self.feat_dim = None
        self.transform = data_utils.imagenet_transform(phase, transform_type)
        self.loader = data_utils.ImageLoader(self.root+'/images/')

        if feat_file is not None:
#           print ("The feature file loaded is " + str(feat_file))
          feat_file = os.path.join(root, feat_file)
          activation_data = torch.load(feat_file)
          activation_data['files'] = ['_'.join(file.split()) for file in activation_data['files']]
          self.activation_dict = dict(zip(activation_data['files'], activation_data['features']))
          self.feat_dim = activation_data['features'].size(1)
          print ('%d activations loaded'%(len(self.activation_dict)))


        # pair = (attr, obj)
        (self.attrs, self.objs, self.pairs, 
        self.train_pairs, self.val_pairs, self.test_pairs) = self.parse_split()

        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}

        self.set_ignore_mode(ignore_attrs, ignore_objs)

        if self.ignore_mode:
          all_pairs = self.train_pairs + self.val_pairs + self.test_pairs
          self.pairs = sorted(list(set(self.train_pairs + self.val_pairs + self.test_pairs)))
          self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
          train_pairs = [(attr, obj) for attr, obj in all_pairs if not self.ignored(attr, obj)]
          val_pairs = [(attr, obj) for attr, obj in all_pairs if self.ignored(attr, obj)]
          test_pairs = [(attr, obj) for attr, obj in all_pairs if self.ignored(attr, obj)]
          self.train_pairs, self.val_pairs, self.test_pairs = train_pairs, val_pairs, test_pairs
          
        self.op_pair2idx = dict()
        for i, attr in enumerate(self.attrs):
          for j, obj in enumerate(self.objs):
            self.op_pair2idx[(attr, obj)] = i * len(self.objs) + j

        self.open_world = open_world
        if open_world:
          self.all_pair2idx = self.op_pair2idx
        else:
          self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        if train_only:
          self.pair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}
        else:
          self.pair2idx = self.all_pair2idx

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        
        if self.phase=='train':
            self.data = self.train_data
        elif self.phase=='val':
            self.data = self.val_data
        elif self.phase=='test':
            self.data = self.test_data
        
        # list of [img_name, attr, obj, attr_id, obj_id, feat]
        print ('#images = %d'%len(self.data))
        
        # Sample imgs with the same obj but different attr
        samples_grouped_by_obj = [[] for _ in range(len(self.objs))]
        for i,x in enumerate(self.data):
            samples_grouped_by_obj[x[4]].append(i)

        self.sample_pool = []  # [obj_id][attr_id] => list of sample id
        for obj_id in range(len(self.objs)):
            self.sample_pool.append([])
            for attr_id in range(len(self.attrs)):
                self.sample_pool[obj_id].append(
                    [i for i in samples_grouped_by_obj[obj_id] if 
                        self.data[i][3] != attr_id]
                )
          
    def set_ignore_mode(self, ignore_attrs, ignore_objs):
        if ignore_attrs and type(ignore_attrs[0]) is str:
          ignore_attrs = [self.attr2idx[attr] for attr in ignore_attrs]
        self.ignore_attrs = set(ignore_attrs)
        if ignore_objs and type(ignore_objs[0]) is str:
          ignore_objs = [self.obj2idx[obj] for obj in ignore_objs]
        self.ignore_objs = set(ignore_objs)
        
        self.ignore_mode = bool(ignore_objs or ignore_attrs)
                
    def ignored(self, attr, obj):
        attr = attr if type(attr) is int else self.attr2idx[attr]
        obj = obj if type(obj) is int else self.obj2idx[obj]
        return self.ignore_mode and (attr in self.ignore_attrs or obj in self.ignore_objs)

    def get_split_info(self):
        data = torch.load(self.root+'/metadata.t7')
        train_pair_set = set(self.train_pairs)
        test_pair_set = set(self.test_pairs)
        train_data, val_data, test_data = [], [], []

        print("natural split "+self.phase)
        for instance in data:
            image, attr, obj, settype = instance['image'], instance['attr'], instance['obj'], instance['set']

            if attr=='NA' or (attr, obj) not in self.pairs or settype=='NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue
            attr_id, obj_id = self.attr2idx[attr], self.obj2idx[obj]  
            data_i = [image, attr, obj, attr_id, obj_id]
            
            if settype == 'train':
                if not self.ignored(attr_id, obj_id):
                    train_data.append(data_i)
                else:
                    if self.phase == 'val':
                      val_data.append(data_i)
                    else:
                      test_data.append(data_i)
            elif settype == 'val':
                if self.ignore_mode and not self.ignored(attr_id, obj_id):
                  train_data.append(data_i)
                elif self.ignore_mode: # for the compatibility with CompAE dataset
                  test_data.append(data_i)
                else:
                  val_data.append(data_i)
                  
            elif settype == 'test':
                if self.ignore_mode and not self.ignored(attr_id, obj_id):
                  train_data.append(data_i)
                else:
                  test_data.append(data_i)
            else:
                raise NotImplementedError(settype)

        return train_data, val_data, test_data


    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list,'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs('%s/%s/train_pairs.txt'%(self.root, self.split))
        val_attrs, val_objs, val_pairs = parse_pairs('%s/%s/val_pairs.txt'%(self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs('%s/%s/test_pairs.txt'%(self.root, self.split))

        all_attrs =  sorted(list(set(tr_attrs + val_attrs + ts_attrs)))
        all_objs = sorted(list(set(tr_objs + val_objs + ts_objs)))    
        all_pairs = sorted(list(set(tr_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, val_pairs, ts_pairs


    def random_sample(self, attr_id, obj_id):
        candidates = self.sample_pool[obj_id][attr_id]
        if len(candidates)==0:
          raise Exception(f"Can't find random sampling candidates for the pair: {self.attrs[attr_id]}, {self.objs[obj_id]}.")
        return np.random.choice(candidates)

    def __getitem__(self, index):
        def get_sample(i):
            image_name, attr, obj, attr_id, obj_id = self.data[i]

            if self.with_image:
                img = self.loader(image_name)
                img = self.transform(img)
            else:
                img = self.activation_dict[image_name]
            return [image_name, attr_id, obj_id, self.pair2idx[(attr, obj)], img]
          
        def get_batch_sample(sample_ids):
          samples = [get_sample(i) for i in sample_ids]
          samples = [list(x) for x in zip(*samples)]
          return samples
        
        sample = get_sample(index)
        attr_id, obj_id = sample[1], sample[2]
        
        if not self.random_sampling:
          rand_sample = []
        else:
          rand_sample_id = self.random_sample(attr_id, obj_id) # negative example
          rand_sample = get_sample(rand_sample_id)

        data = sample + rand_sample

        # train [img_name, attr_id, obj_id, pair_id, img_feature, img_name, attr_id, obj_id, pair_id, img_feature]

        return data

    def __len__(self):
        return len(self.data)


class CompositionDatasetActivationsGenerator(CompositionDatasetActivations):

    def __init__(self, root, feat_file, split='compositional-split', feat_extractor=None, transform_type='normal'):
        super(CompositionDatasetActivationsGenerator, self).__init__(root, 'train', None, split, transform_type=transform_type)

        assert os.path.exists(root)
        with torch.no_grad():
            self.generate_features(feat_file, feat_extractor, transform_type)
        print('Features generated.')

    def get_split_info(self):
        data = torch.load(self.root+'/metadata.t7')
        train_pair_set = set(self.train_pairs)
        test_pair_set = set(self.test_pairs)
        train_data, val_data, test_data = [], [], []

        print("natural split")
        for instance in data:
            image, attr, obj, settype = instance['image'], instance['attr'], instance['obj'], instance['set']

            if attr=='NA' or (attr, obj) not in self.pairs or settype=='NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue
                
            data_i = [image, attr, obj, self.attr2idx[attr], self.obj2idx[obj], None]

            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            elif settype == 'test':
                test_data.append(data_i)
            else:
                raise NotImplementedError(settype)

        return train_data, val_data, test_data
        

    def generate_features(self, out_file, feat_extractor, transform_type):

        data = self.train_data+self.val_data+self.test_data
        transform = data_utils.imagenet_transform('test', transform_type)

        if feat_extractor is None:
            feat_extractor = torchvision.models.resnet18(pretrained=True)
            feat_extractor.fc = torch.nn.Sequential()
        feat_extractor.eval().cuda()

        image_feats = []
        image_files = []
        for chunk in tqdm.tqdm(data_utils.chunks(data, 512), total=len(data)//512):
            files = zip(*chunk)[0]
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs))
            feats = feat_extractor(torch.stack(imgs, 0).cuda())
            image_feats.append(feats.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, 0)
        print ('features for %d images generated'%(len(image_files)))

        torch.save({'features': image_feats, 'files': image_files}, out_file)



if __name__=='__main__':
    """example code for generating new features for MIT states and UT Zappos
    CompositionDatasetActivationsGenerator(
        root = 'data-dir', 
        feat_file = 'filename-to-save', 
        feat_extractor = torchvision.models.resnet18(pretrained=True),
    )
    """

    if sys.argv[1]=="MIT":
        name = "mit-states"
    elif sys.argv[1]=="UT":
        name = "ut-zap50k"
    

    CompositionDatasetActivationsGenerator(
        root = 'data/%s-natural'%name, 
        feat_file = 'data/%s-natural/features.t7'%name,
    )
