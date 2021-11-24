import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn
from torch_geometric.utils import negative_sampling

from scipy.sparse import coo_matrix
import numpy as np

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

import torchvision.models as models

cross_entropy_loss = nn.CrossEntropyLoss()

resnet_models = {'resnet18': models.resnet18,
                 'resnet50': models.resnet50,
                 'resnet101': models.resnet101,
                 'resnet152': models.resnet152}


def frozen(model):
  for param in model.parameters():
    param.requires_grad = False
  return model
  
  
class ParametricMLP(nn.Module):
  '''Output size of each inner layer specified by [layer_sizes]'''
  def __init__(self, in_features, out_features, layer_sizes, batch_norm=True, norm_output=False, dropout=0.5):
    super(ParametricMLP, self).__init__()
    layers = []
    for layer_size in layer_sizes:
      layer = nn.Sequential(
        nn.Linear(in_features, layer_size),
        nn.BatchNorm1d(layer_size) if batch_norm else nn.Identity(),
        nn.ReLU(),
        nn.Dropout(dropout))
      layers.append(layer)
      in_features = layer_size
    layers.append(nn.Linear(in_features, out_features))
    if norm_output:
      layers.append(nn.LayerNorm(out_features, elementwise_affine=False))
    self.mlp = nn.Sequential(*layers)
    
  def forward(self, x):
    return self.mlp(x)

class GraphModelBase(nn.Module):
  def __init__(self, hparam, dset, graph_path, train_only=False, resnet_name=None, static_inp=True):
    super(GraphModelBase, self).__init__()
    self.hparam = hparam
    self.nattrs, self.nobjs = len(dset.attrs), len(dset.objs)
    self.img_feat_dim = dset.feat_dim

    if resnet_name:
      assert resnet_name in resnet_models, f"{resnet_name} doesn't match any known models"
      self.resnet = resnet_models[resnet_name](pretrained=True).to(dev)
      if static_inp:
        self.resnet = frozen(self.resnet)
      self.img_feat_dim = self.resnet.fc.in_features
      self.resnet.fc = nn.Identity()
    else:
      self.resnet = None
        
    self.train_only = train_only
    if self.train_only:
      train_idx = []
      for pair in dset.train_pairs:
        train_idx.append(dset.all_pair2idx[pair])
      self.train_idx = torch.LongTensor(train_idx).to(dev)

    graph = torch.load(graph_path)
    self.nodes = graph["embeddings"].to(dev)
    adj = graph["adj"]
    self.edge_index = torch.tensor(np.array([adj.row, adj.col]), dtype=torch.long).to(dev) # np.array purely for suppressing pytorch conversion warning
      
      
class VariationalEncoder(nn.Module):
    def __init__(self, conv_layer, in_features, out_features, hidden_layer_sizes=[]):
      super(VariationalEncoder, self).__init__()
      orig_in_features = in_features
      hidden_layers = []
      for hidden_size in hidden_layer_sizes:
        layer = [
          (conv_layer(in_features, hidden_size), 'x, edge_index -> x'),
          nn.ReLU(),
          nn.Dropout()
        ]
        hidden_layers.extend(layer)
        in_features = hidden_size
      hidden_layers.append((conv_layer(in_features, out_features), 'x, edge_index -> x'))

      self.encodermu = gnn.Sequential('x, edge_index', hidden_layers)
      
      varhidden_layers = []
      for hidden_size in hidden_layer_sizes:
        varlayer = [
          (conv_layer(orig_in_features, hidden_size), 'x, edge_index -> x'),
          nn.ReLU(),
          nn.Dropout()
        ]
        varhidden_layers.extend(varlayer)
        orig_in_features = hidden_size
      varhidden_layers.append((conv_layer(in_features, out_features), 'x, edge_index -> x'))

      self.encodervar = gnn.Sequential('x, edge_index', varhidden_layers)

    def forward(self, x, edge_index):
        return self.encodermu(x, edge_index), self.encodervar(x, edge_index)
      


def recon_loss(model, z):
  """Only consider valid edges (attr-obj) when calculating reconstruction loss"""
  from torch_geometric.utils import remove_self_loops, add_self_loops
  from utils import gae_negative_sampling as negative_sampling
  
  pos_edge_index = model.train_pair_edges
  pos_loss = -torch.log(
      model.gae.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

  # Do not include self-loops in negative samples

  pos_edge_index, _ = remove_self_loops(pos_edge_index)
  pos_edge_index, _ = add_self_loops(pos_edge_index)
  neg_edge_index = negative_sampling(pos_edge_index, model.nattrs, model.nobjs)
  neg_loss = -torch.log(1 -
                        model.gae.decoder(z, neg_edge_index, sigmoid=True) +
                        1e-15).mean()

  return pos_loss + neg_loss
  
   
class CVGAE(gnn.VGAE):
    def __init__(self, hparam, dset, graph_path=None, train_only=False, resnet_name=None):
        
        self.hparam = hparam
        self.nattrs, self.nobjs = len(dset.attrs), len(dset.objs)
        self.img_feat_dim = dset.feat_dim
        self.num_nodes = self.nattrs + self.nobjs
        self.norm_factor = self.num_nodes * self.num_nodes
        
        self.dset = dset
        self.train_only =train_only
        
        if self.train_only:
          train_idx = []
          for pair in dset.train_pairs:
            train_idx.append(dset.all_pair2idx[pair])
          self.train_idx = torch.LongTensor(train_idx).to(dev)
        
        graph = torch.load(graph_path)
        self.nodes = graph["embeddings"][:self.nattrs+self.nobjs].to(dev)
        adj = graph["adj"].todense()
        # This makes sure that the graph contains only primitives concept
        adj = adj[:self.nattrs+self.nobjs, :self.nattrs+self.nobjs]

        # Initialized the fscore
        self.fscore = torch.tensor(adj[:self.nattrs,self.nattrs:]).to(dev)        

        adj = coo_matrix(adj)
        self.node_dim = 150
        if not self.dset.name == 'CGQA':
          self.hparam.add('shared_emb_dim', 1500)
          self.hparam.add_dict({'graph_encoder_layers': [2048], 'node_dim': 150})
        else: # Due to memory issue we use less dimensions for CGQA
          self.hparam.add('shared_emb_dim', 800)
          self.hparam.add_dict({'graph_encoder_layers': [1024], 'node_dim': 150})

        encoder = VariationalEncoder(gnn.SAGEConv, self.nodes.size(1), self.hparam.node_dim, self.hparam.graph_encoder_layers)
        super(CVGAE, self).__init__(encoder)
        self.loss_ie = 0
        row_idx, col_idx = adj.row, adj.col
        self.train_pair_edges = torch.tensor(np.array([row_idx, col_idx]), dtype=torch.long).to(dev)

        # We load from disk the extraced Resnet-18 features.
        # They are the same features as CompCos, CGE, for fair comparison.
        # We do not report results for trainable feature extractor (ResNet etc)
        # This is to ensure fairness with the baseline models
        # But if one wishes then one can employ trainable feature extractor by using 
        # the following lines of code
        
        # Loading Resnet 18 model
        if resnet_name:    
          import torchvision.models as models
          self.resnet = models.resnet18(pretrained=True).to(dev)
          self.img_feat_dim = self.resnet.fc.in_features
          self.resnet.fc = nn.Linear(self.img_feat_dim, self.img_feat_dim)
        else:
          self.resnet = None
          # Default is to use same features as employed by CompCos, CGE, for fair comparison.
          # They are saved on the disk by using thier code.

         # Updating of initial node embeddings is set as False; minor improv. expected if set to True.
        update_emb = False
        if update_emb:
          self.nodes = nn.Parameter(self.nodes)
        
        
        self.hparam.add_dict({'img_fc_layers': [800, 1000], 'img_fc_norm': True,
                          'pair_fc_layers': [1000], 'pair_fc_norm': True})
        self.img_fc = ParametricMLP(self.img_feat_dim, self.hparam.shared_emb_dim, self.hparam.img_fc_layers,
                                norm_output=self.hparam.img_fc_norm)
        self.pair_fc = ParametricMLP(self.hparam.node_dim*2, self.hparam.shared_emb_dim, 
                                 self.hparam.pair_fc_layers, norm_output=self.hparam.pair_fc_norm)
            
    def get_all_pairs(self, nodes):
      attr_nodes = nodes[:self.nattrs]
      obj_nodes = nodes[self.nattrs:]
      if self.dset.open_world:
        all_pair_attrs = attr_nodes.repeat(1,self.nobjs).view(-1, self.node_dim)
        all_pair_objs = obj_nodes.repeat(self.nattrs, 1)
      else:
        pairs = self.dset.pairs
        all_pair_attr_ids = [self.dset.attr2idx[attr] for attr, obj in pairs]
        all_pair_obj_ids = [self.dset.obj2idx[obj] for attr, obj in pairs]
        all_pair_attrs = attr_nodes[all_pair_attr_ids]
        all_pair_objs = obj_nodes[all_pair_obj_ids]
        
      all_pairs = torch.cat((all_pair_attrs, all_pair_objs), dim=1)
      if self.train_only and self.training:
        all_pairs = all_pairs[self.train_idx]
      return all_pairs

    def forward(self, x):
      if self.resnet:
        img = self.resnet(x[4].to(dev))
      else:
        img = x[4].to(dev)
      
      img_feats = self.img_fc(img)
      z_nodes = self.encode(self.nodes, self.train_pair_edges)
      z_attrs, z_objs = z_nodes[:self.nattrs], z_nodes[self.nattrs:]
      link_pred = torch.matmul(z_attrs, z_objs.T)
      link_probs = torch.softmax(link_pred, dim=1)
      link_probs = F.normalize(link_probs,dim=1)
      with torch.no_grad(): # to avoid the gradient and memory overflow
        self.fscore +=  link_probs / self.norm_factor
      
      all_pair_nodes = self.get_all_pairs(z_nodes)
      all_pairs = self.pair_fc(all_pair_nodes)

      # L_e-->i Approach
      pair_pred = torch.matmul(img_feats, all_pairs.T)
      
      # L_i-->e Approach
      # second view BBCE Loss
      pair_labels = x[3].to(dev)
      true_pairs = all_pairs[pair_labels]
      batch_sim_ie = torch.mm(true_pairs, img_feats.transpose(0, 1))
      labels = torch.tensor(range(batch_sim_ie.shape[0])).long()
      labels = torch.autograd.Variable(labels).cuda()
      
      self.loss_ie = cross_entropy_loss(batch_sim_ie, labels) / self.norm_factor 
      return pair_pred, z_nodes, self



def cvgae_loss(model_output, sample):
  pair_pred, nodes, model = model_output
  recon_loss = model.recon_loss(nodes, model.train_pair_edges)
  kl_loss = (1 / model.norm_factor) * model.kl_loss()

  pair_labels = sample[3].to(dev)
  loss_ei = cross_entropy_loss(pair_pred, pair_labels)

  loss_ie = model.loss_ie
  lambda_ei = 10
  lambda_ie = 0.01
  total_loss = lambda_ei * loss_ei + lambda_ie * loss_ie + recon_loss + kl_loss
  loss_dict = {'loss_ei': loss_ei, 'recon_loss':recon_loss, 'kl_loss':kl_loss, 'loss_ie': loss_ie}
  return total_loss, loss_dict