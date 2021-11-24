import dataset
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

class DummyLogger():
  log_dir = None
  filename_suffix = None
  def add_scalar(self, a, b, c):
    pass
  def add_hparams(self, a, b):
    pass
  def add_text(self, a, b):
    pass
  def flush(self):
    pass
  def close(self):
    pass
  
class HParam():
  def __init__(self):
    self.hparam_dict = dict() # For tensorboard logger and restoring state
    self.frozen = False # If true, attributes can't be updated. But new attribute can still be added
    
  def add(self, name, value):
    if self.frozen and name in self.hparam_dict:
      return
    setattr(self, name, value)
    self.hparam_dict[name] = value
      
  def add_dict(self, d):
    for name, value in d.items():
      self.add(name, value)
      
  def get(self, name):
    if name not in self.hparam_dict:
      raise Exception(f"HParam: {name} doesn't exist!")
    return getattr(self, name)
  
  def freeze(self):
    self.frozen = True
  
  def __repr__(self):
    return repr(self.hparam_dict)
  
cross_entropy_loss = nn.CrossEntropyLoss()

class _Loss():
  name = ''
  def __init__(self):
    self.hparam_dict = {'name': self.name}
    
  def __call__(self, model_output, sample):
    return self.loss(model_output, sample)
  

class DummyLoss(_Loss):
  name = 'DummyLoss'
  def loss(self, model_output, sample):
    return None, dict()
  
dummy_loss = DummyLoss()
  
class EuclideanLoss(_Loss):
  name = 'EuclideanLoss'
  def loss(self, model_output, sample):
    x, y = model_output # [batch_size, npairs]
    loss = F.pairwise_distance(x, y).mean()
    loss_dict = {'ed_loss': loss}
    return loss, loss_dict
  
euclidean_dist_loss = EuclideanLoss()
  
class PrimitiveCE(_Loss):
  name = 'PrimitiveCE'
  def loss(self, model_output, sample):
    attr_scores, obj_scores = model_output
    attr_labels = sample[1].to(dev)
    obj_labels = sample[2].to(dev)
    attr_loss = cross_entropy_loss(attr_scores, attr_labels)
    obj_loss = cross_entropy_loss(obj_scores, obj_labels)
    loss_dict = {'attr_loss': attr_loss, 'obj_loss': obj_loss}
    total_loss = attr_loss + obj_loss
    return total_loss, loss_dict
  
primitive_cross_entropy_loss = PrimitiveCE()

class PairCE(_Loss):
  name = 'PairCE'
  def loss(self, model_output, sample):
    compo_score = model_output # [batch_size, npairs]
    pair_labels = sample[3].to(dev)
    loss = cross_entropy_loss(compo_score, pair_labels)
    loss_dict = {'contra_loss': loss}
    return loss, loss_dict
  
pair_cross_entropy_loss = PairCE()

class BatchCE(_Loss):
  name = 'BatchCE'
  def loss(self, model_output, sample):
    embs, targets = model_output # [batch_size, npairs]
    nsample = len(embs)
    scores = embs @ targets.T
    labels = torch.tensor(range(nsample)).to(dev)
    loss = cross_entropy_loss(scores, labels)
    loss_dict = {'batch_ce_loss': loss}
    return loss, loss_dict
  
batch_ce_loss = BatchCE()


class IR_loss(_Loss):
  name = 'IR_loss'
  def loss(self, model_output, sample):
    embs, targets, model = model_output # [batch_size, npairs]
    z_nodes = model.encode(model.nodes, model.train_pair_edges)
    recon_loss = model.recon_loss(z_nodes, model.train_pair_edges)
    kl_loss = (1 / model.norm_factor) * model.kl_loss()

    nsample = len(embs)
    scores = embs @ targets.T
    labels = torch.tensor(range(nsample)).to(dev)
    loss = cross_entropy_loss(scores, labels)
    
    total_loss = loss +  recon_loss + kl_loss
    loss_dict = {'loss': loss, 'recon_loss':recon_loss, 'kl_loss':kl_loss}
    return total_loss, loss_dict
  
cvgae_loss_IR = IR_loss()



class ObjAwareLoss(_Loss):
  name = 'ObjAwareLoss'
  def loss(self, model_output, sample):
    theta, targets, neg_pairs = model_output
    nsample = len(theta)
    loss = 0
    for i in range(nsample):
      loss += torch.log(1+torch.exp((neg_pairs[i]@targets[i].T - theta[i]@targets[i].T)/2)).mean()
    loss /= nsample
    simple_loss, _ = batch_ce_loss((theta, targets), sample)
    total_loss = loss + simple_loss
    loss_dict = {'batch_ce_loss': simple_loss, 'obj_aware_loss': loss}
    return total_loss, loss_dict
  
obj_aware_loss = ObjAwareLoss()



class GAELoss(_Loss):
  name = 'GAELoss'
  def __init__(self, recon_loss_ratio):
    super().__init__();
    self.recon_loss_ratio = recon_loss_ratio
    
  def loss(self, model_output, sample):
    pair_pred, nodes, model = model_output
    recon_loss = model.gae.recon_loss(model, nodes)
    ce_loss, ce_loss_dict = pair_cross_entropy_loss(pair_pred, sample)
    total_loss = (1-self.recon_loss_ratio) * ce_loss + self.recon_loss_ratio * recon_loss
    loss_dict = {'recon_loss':recon_loss} | ce_loss_dict
    return total_loss, loss_dict
  
  
class MetricLearningLoss(_Loss):
  name = 'MetricLearningLoss'
  def __init__(self, ml_loss, loss_weights=[1, 1], miner=None):
    super().__init__()
    self.ml_loss = ml_loss
    self.miner = miner
    self.img_loss_weight, self.text_loss_weight = loss_weights
    self.hparam_dict |= {'ml_loss_func': self.ml_loss.__class__.__name__,
           'ml_loss_miner': self.miner.__class__.__name__ if self.miner else 'None'}
    
  def loss(self, model_output, sample):
    img_feats, pair_emb = model_output
    pair_id = sample[3]
    img_miner_output = self.miner(img_feats, pair_id) if self.miner else None
    img_loss = self.ml_loss(img_feats, pair_id, indices_tuple=img_miner_output)
    text_miner_output = self.miner(pair_emb, pair_id) if self.miner else None
    text_loss = self.ml_loss(pair_emb, pair_id, indices_tuple=text_miner_output)
    total_loss = self.img_loss_weight * img_loss + self.text_loss_weight * text_loss
    loss_dict = {'img_loss': img_loss, 'text_loss': text_loss}
    return total_loss, loss_dict


class GAEMLLoss(_Loss):
  name = 'GAEMLLoss'
  def __init__(self, loss_func, loss_weights, miner=None):
    super().__init__()
    self.ml_weights = loss_weights[:2]
    self.primitive_weight, self.ce_weight, self.recon_weight = loss_weights[2:]
    self.ml_loss = MetricLearningLoss(loss_func, self.ml_weights, miner)
    self.hparam_dict = self.ml_loss.hparam_dict | self.hparam_dict
    
  def loss(self, model_output, sample):    
    pair_pred, attr_pred, obj_pred, img_feats, pair_emb, nodes, model = model_output  
    primitive_loss, primitive_loss_dict = primitive_cross_entropy_loss((attr_pred, obj_pred), sample)
    ce_loss, ce_loss_dict = pair_cross_entropy_loss(pair_pred, sample)
    ml_total_loss, ml_loss_dict = self.ml_loss((img_feats, pair_emb), sample)
    recon_loss = model.gae.recon_loss(nodes, model.train_pair_edges)
    total_loss = ml_total_loss + self.primitive_weight * primitive_loss + self.ce_weight * ce_loss + self.recon_weight * recon_loss
    loss_dict = {'recon_loss': recon_loss} | primitive_loss_dict | ce_loss_dict | ml_loss_dict 
    return total_loss, loss_dict

  def hparam_dict(self):
    return self.ml_loss.hparam_dict() | {'loss_ratio': self.ml_weights + [self.primitive_weight, self.ce_weight, self.recon_weight]}
  

class GAEEDLoss(_Loss):
  name = 'GAEEDLoss'
  def __init__(self, loss_func, loss_weights, miner=None):
    super().__init__()
    self.ml_weights = loss_weights[:2]
    self.primitive_weight, self.pair_weight, self.recon_weight = loss_weights[2:]
    self.ml_loss = MetricLearningLoss(loss_func, self.ml_weights, miner)
    self.hparam_dict = self.ml_loss.hparam_dict | self.hparam_dict
    
  def loss(self, model_output, sample):
    if not isinstance(model_output, tuple):
      return pair_cross_entropy_loss(model_output, sample)
    attr_pred, obj_pred, img_feats, pair_emb, nodes, model = model_output
    primitive_loss, primitive_loss_dict = primitive_cross_entropy_loss((attr_pred, obj_pred), sample)
    ed_loss, ed_loss_dict = euclidean_dist_loss((img_feats, pair_emb), sample)
    ml_total_loss, ml_loss_dict = self.ml_loss((img_feats, pair_emb), sample)
    recon_loss = model.gae.recon_loss(nodes, model.train_pair_edges)
    total_loss = ml_total_loss + self.primitive_weight * primitive_loss + self.pair_weight * ed_loss + self.recon_weight * recon_loss
    loss_dict = {'recon_loss': recon_loss} | primitive_loss_dict | ed_loss_dict | ml_loss_dict 
    return total_loss, loss_dict


class ReciprocalLoss(_Loss):
  name = 'ReciprocalLoss'
  def __init__(pre_loss_scale=1, adaptive_scale=False, total_epochs=None):
    super().__init__()
    self.pre_loss_scale = pre_loss_scale
    assert(not adaptive_scale or total_epochs)
    self.epoch = 0
    self.total_epochs = total_epochs
    self.adaptive_scale = adaptive_scale

  def loss(self, model_output, sample):
    attr_scores, obj_scores, attr_pre_scores, obj_pre_scores = model_output
    attr_labels = sample[1].to(dev)
    obj_labels = sample[2].to(dev)
    attr_loss = cross_entropy_loss(attr_scores, attr_labels)
    attr_pre_loss = cross_entropy_loss(attr_pre_scores, attr_labels)
    obj_loss = cross_entropy_loss(obj_scores, obj_labels)
    obj_pre_loss = cross_entropy_loss(obj_pre_scores, obj_labels)
    loss_dict = {'attr_loss': attr_loss, 'obj_loss': obj_loss, 'attr_pre_loss': attr_pre_loss, 'obj_pre_loss': obj_pre_loss}
    if self.adaptive_scale:
      # reduce pre_loss_scale linearly to 0 in the middle of training
      pre_loss_scale = max(0, 1-self.epoch/(self.total_epochs/2))
    total_loss = (1-self.pre_loss_scale) * (attr_loss + obj_loss) + self.pre_loss_scale * (attr_pre_loss + obj_pre_loss)
    self.epoch += 1
    return total_loss, loss_dict
  

class NPairLoss(_Loss):
  name = 'NPairLoss'
  def __init__(self, margin=0.1):
    super().__init__()
    self.margin = margin
    
  def loss(self, model_output, sample):
    embs, anchors = model_output
    nsample = len(embs)
    pair_id = sample[3]
    dot = embs @ anchors.T
    pos = dot.diag() # dist of correct pairs of (img, label)
#     loss = torch.log(torch.exp(dot - pos.view(1, -1)).sum(dim=0)).mean()
    loss = dot - pos.view(1, -1) + self.margin
    loss = torch.max(loss, torch.zeros_like(loss)).sum(dim=0).mean()
    loss_dict = {'npair_loss': loss}
    return loss, loss_dict
  
npair_loss = NPairLoss()
  
class GAENPLoss(_Loss):
  name = 'GAENPLoss'
  def __init__(self, loss_weights):
    super().__init__()
    self.img_npair_weight, self.node_npair_weight, self.primitive_weight, self.ce_weight, self.recon_weight = loss_weights
    
  def loss(self, model_output, sample):    
    pair_pred, attr_pred, obj_pred, img_feats, pair_emb, nodes, model = model_output  
    primitive_loss, primitive_loss_dict = primitive_cross_entropy_loss((attr_pred, obj_pred), sample)
    ce_loss, ce_loss_dict = pair_cross_entropy_loss(pair_pred, sample)
    img_npair_loss, img_npair_loss_dict = npair_loss((img_feats, pair_emb), sample)
    node_npair_loss, node_npair_loss_dict = npair_loss((pair_emb, img_feats), sample)
    recon_loss = model.gae.recon_loss(nodes, model.train_pair_edges)
    total_loss = self.img_npair_weight * img_npair_loss + self.node_npair_weight * node_npair_loss + \
                + self.primitive_weight * primitive_loss + self.ce_weight * ce_loss + self.recon_weight * recon_loss
    loss_dict = {'recon_loss': recon_loss, 'img_npair_loss': img_npair_loss, 'node_npair_loss': node_npair_loss} | primitive_loss_dict | ce_loss_dict
    return total_loss, loss_dict

  def hparam_dict(self):
    return self.ml_loss.hparam_dict() | {'loss_ratio': self.ml_weights + [self.primitive_weight, self.ce_weight, self.recon_weight]}


class EuclidNpairLoss(_Loss):
  name = 'EuclidNpairLoss'
  def __init__(self, margin=0.1):
    super().__init__()
    self.margin = margin
    
  def loss(self, model_output, sample):
    if not isinstance(model_output, tuple):
      return pair_cross_entropy_loss(model_output, sample)
    img_feats, all_pairs = model_output
    nsample = len(img_feats)
    pair_id = sample[3]
    dist = torch.cdist(img_feats, all_pairs)
    s_it = dist[range(nsample), pair_id] # dist of correct pairs of (img, label)
    negative = dist[:, pair_id]
    s_nit = s_it.view(1, -1) - negative
    s_int = s_it.view(-1, 1) - negative
    loss = torch.max(self.margin+s_int, torch.zeros_like(s_int)).sum(dim=1).mean()
    loss_dict = {'ed_npair_loss': loss}
    return loss, loss_dict


def generate_graph_op(dataset_name):
  from scipy.sparse import coo_matrix
  w2v_attrs = torch.load(os.path.join('./embeddings', dataset_name, 'w2v_attrs.pt'))
  w2v_objs = torch.load(os.path.join('./embeddings', dataset_name, 'w2v_objs.pt'))
  nattrs = len(w2v_attrs)
  nobjs = len(w2v_objs)
  ind = []
  values = []
  adj_dim = nattrs + nobjs + nattrs*nobjs
  for i in range(nattrs):
    for j in range(nobjs):
      ind.append([i, nattrs + j]) # attr -> obj
      ind.append([i, i*nobjs+j+nattrs+nobjs]) # attr -> compo
      ind.append([nattrs+j, i*nobjs+j+nattrs+nobjs]) # obj -> compo
      
  back_ward = [(j, i) for i, j in ind]
  ind += back_ward
  for i in range(adj_dim):
    ind.append((i,i))

  ind = torch.tensor(ind)    
  values = torch.tensor([1] * len(ind), dtype=torch.float)
  adj = coo_matrix((values, (ind[:, 0].numpy(), ind[:, 1].numpy())))
  
  embs = []
  for i in range(nattrs):
    embs.append(w2v_attrs[i])
  for i in range(nobjs):
    embs.append(w2v_objs[i])
  for i in range(nattrs):
    for j in range(nobjs):
      compo_emb = (w2v_attrs[i] + w2v_objs[j]) / 2
      embs.append(compo_emb)

  embs = torch.vstack(embs)
  
  return {"adj": adj, "embeddings": embs}

def generate_graph_primitive(dataset_name):
  from scipy.sparse import coo_matrix
  w2v_attrs = torch.load(os.path.join('./embeddings', dataset_name, 'w2v_attrs.pt'))
  w2v_objs = torch.load(os.path.join('./embeddings', dataset_name, 'w2v_objs.pt'))
  nattrs = len(w2v_attrs)
  nobjs = len(w2v_objs)
  ind = []
  values = []
  adj_dim = nattrs + nobjs
  for i in range(nattrs):
    for j in range(nobjs):
      ind.append([i, nattrs+j]) # attr -> obj
      ind.append([nattrs+j, i]) # obj -> attr

  for i in range(adj_dim):
    ind.append((i,i))

  ind = torch.tensor(ind)    
  values = torch.tensor([1] * len(ind), dtype=torch.float)
  adj = coo_matrix((values, (ind[:, 0].numpy(), ind[:, 1].numpy())))
  
  embs = []
  for i in range(nattrs):
    embs.append(w2v_attrs[i])
  for i in range(nobjs):
    embs.append(w2v_objs[i])

  embs = torch.vstack(embs)
  
  return {"adj": adj, "embeddings": embs}

def gae_negative_sampling(edge_index, nattrs, nobjs, num_nodes=None, num_neg_samples=None):
  from torch_geometric.utils.num_nodes import maybe_num_nodes
  def sample(idx, size, device=None):
    size = min(len(idx), size)
    return torch.tensor(np.random.choice(idx, size), device=dev)
  
  num_nodes = nattrs + nobjs
  num_neg_samples = num_neg_samples or edge_index.size(1)

  # Handle '|V|^2 - |E| < |E|'.
  size = num_nodes * num_nodes
  num_neg_samples = min(num_neg_samples, size - edge_index.size(1))

  row, col = edge_index
  idx = row * num_nodes + col
  all_pair_row, all_pair_col = np.meshgrid(range(nattrs), range(nattrs, nobjs+nattrs))
  all_pair_idx = all_pair_row.ravel() * num_nodes + all_pair_col.ravel()

  # Percentage of edges to oversample so that we are save to only sample once
  # (in most cases).
  alpha = abs(1 / (1 - 1.2 * (edge_index.size(1) / size)))

  perm = sample(all_pair_idx, int(alpha * num_neg_samples))
  mask = torch.from_numpy(np.isin(perm.to('cpu'), idx.to('cpu'))).to(torch.bool)
  perm = perm[~mask][:num_neg_samples].to(edge_index.device)
  
  row = torch.div(perm, num_nodes, rounding_mode='floor')
  col = perm % num_nodes
  neg_edge_index = torch.stack([row, col], dim=0).long()

  return neg_edge_index