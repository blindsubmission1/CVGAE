import torch
import torch.nn.functional as F
import numpy as np
from gensim.models import KeyedVectors
import tqdm

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

class _BaseEvaluator():
  def __init__(self, test_dataloader, num_bias, cpu_eval=False):
    self.num_bias = num_bias
    self.test_dataloader = test_dataloader
    self.attrs, self.objs = np.array(test_dataloader.dataset.attrs), np.array(test_dataloader.dataset.objs)
    self.attr_class, self.obj_class = len(self.attrs), len(self.objs)
    self.dev = 'cpu' if cpu_eval else dev
    
    self.test_mask, self.seen_mask = self.getCompoMask(test_dataloader) # (attr x obj) matrices
    self.close_mask = self.test_mask + self.seen_mask
    self.unseen_mask_ow = ~self.seen_mask.to(self.dev) # mask of compositions not seen during training in the open world setting
    self.unseen_mask_cw = self.close_mask.to(self.dev) * self.unseen_mask_ow # mask of compositions not seen during training in the closed world setting
    
    self.no_bias = False
    self.preset_bias = False
    
  def pairId2primitiveId(self, pairId):
    obj_id = pairId % self.obj_class
    attr_id = torch.floor(torch.div(pairId,self.obj_class))
#     attr_id = pairId.div(self.obj_class, rounding_mode='floor')
    return attr_id, obj_id
  
  def getCompoMask(self, dataloader):
    """Mask of (attr x obj) matrix with compositions appeared in the dataset being marked as 1."""
    dset = dataloader.dataset
    obj2idx, attr2idx = dset.obj2idx, dset.attr2idx
    attr_class, obj_class = len(dset.attrs), len(dset.objs)

    train_pairs = dset.train_pairs
    phase2pair = {'train':dset.train_pairs, 'val':dset.val_pairs, 'test':dset.test_pairs}
    test_pairs = phase2pair[dset.phase]

    train_pair_idx = np.array([(attr2idx[attr], obj2idx[obj]) for attr, obj in train_pairs])
    test_pair_idx = np.array([(attr2idx[attr], obj2idx[obj]) for attr, obj in test_pairs])
    test_mask = torch.zeros((attr_class, obj_class), dtype=torch.bool)
    seen_mask = torch.zeros_like(test_mask)
    test_mask[(test_pair_idx[:, 0], test_pair_idx[:, 1])] = True
    seen_mask[(train_pair_idx[:, 0], train_pair_idx[:, 1])] = True
    return test_mask, seen_mask

  def acc(self, preds, labels):
    """Calculate top-k accuracy"""
    if len(preds.shape) == 1:
      preds = preds.unsqueeze(-1)
    if len(labels.shape) == 1:
      labels = labels.unsqueeze(-1)
    match = torch.any(preds == labels, dim=1).float()
    return torch.mean(match)
  
  def preset_biaslist(self, biaslist):
    self.preset_bias = True
    self.biaslist = biaslist

  def get_biaslist(self, compo_scores):
    if self.no_bias:
      return [0]
    nsample = len(compo_scores)
    biased_scores = compo_scores.clone()
#     biased_scores[:,~self.close_mask] = -1e10
    biased_scores += self.unseen_mask_cw * 1e3
    biased_correct_pred_mask = (torch.argmax(biased_scores.reshape(nsample, -1),-1)
                               == self.attr_labels * self.obj_class + self.obj_labels)
    
    if not biased_correct_pred_mask.any():
      return [0]

    compo_scores = compo_scores[biased_correct_pred_mask]
    attr_labels, obj_labels = self.attr_labels[biased_correct_pred_mask], self.obj_labels[biased_correct_pred_mask]
    nsample = len(compo_scores)
    
    scores_correct_pred = compo_scores[range(nsample), attr_labels, obj_labels]
    seen_scores = compo_scores.reshape(nsample, -1)[:,self.seen_mask.reshape(-1)]
    max_seen_scores, _ = torch.max(seen_scores, 1)
    score_diff, _ = torch.sort(max_seen_scores - scores_correct_pred - 1e-4)


    bias_skip = max(len(score_diff) // self.num_bias, 1)
    biaslist = score_diff[::bias_skip]
    
    return sorted(biaslist.cpu().tolist()+[0])
  
  def compo_acc(self, compo_scores, topk=1):
    """Calculate match count lists for each bias term for seen and unseen.
    Return: [2 x biaslist_size] with first row for seen and second row for unseen.
    """

    def _compo_match(compo_scores, obj_labels, attr_labels, topk=1):
      """compo_scores: [batch, attr_class, obj_class]
      Return the count of correct composition predictions.
      """
      ncol = compo_scores.shape[-1]
      _, topk_preds = torch.topk(compo_scores.view(len(compo_scores), -1), topk, dim=-1) # [batch, k]
      topk_attr_preds, topk_obj_preds = self.pairId2primitiveId(topk_preds)
      compo_match = (obj_labels.unsqueeze(1) == topk_obj_preds) * (attr_labels.unsqueeze(1) == topk_attr_preds)
      compo_match = torch.any(compo_match, dim=-1)
      return compo_match
    
    compo_scores_original = compo_scores.clone()

    if not self.open_world:
      compo_scores_original[:,~self.close_mask] = -1e10
    else:
      compo_scores_original[:, self.fscore_mask] = -1e10

    if self.preset_bias:
      biaslist = self.biaslist
    else:
      biaslist = self.get_biaslist(compo_scores_original)
      self.biaslist = biaslist
    
    results = torch.zeros((2, len(biaslist))).to(self.dev)
    target_label_seen_mask = self.seen_mask[self.attr_labels, self.obj_labels]

    for i, bias in enumerate(biaslist):
      if self.open_world:
        compo_scores = compo_scores_original + self.unseen_mask_ow * bias
      else:
        compo_scores = compo_scores_original + self.unseen_mask_cw * bias
      matches = _compo_match(compo_scores, self.obj_labels, self.attr_labels, topk)
      results[0, i] = torch.sum(matches[target_label_seen_mask])
      results[1, i] = torch.sum(matches[~target_label_seen_mask])
    acc = torch.max(results[0] + results[1]) / len(compo_scores_original)
   
    results[0] /= torch.sum(target_label_seen_mask)
    results[1] /= torch.sum(~target_label_seen_mask)
    results = [result.cpu() for result in results]
    return acc, results, biaslist

  def analyse_acc_report(self, acc_table, biaslist):
    """acc_table: [2 x biaslist_size] with first row for seen and second row for unseen.
    Return: best_seen, best_unseen, best_harmonic, auc
    """
    seen, unseen = acc_table[0].cpu(), acc_table[1].cpu()
    best_seen = torch.max(seen)
    best_unseen = torch.max(unseen)
    best_harmonic, loc = torch.topk((seen * unseen) ** (1/2), k=1)
    auc = np.trapz(seen, unseen)
    bias = biaslist[loc]
    return {'Seen': best_seen.item(),
           'Unseen': best_unseen.item(),
           'HM': best_harmonic.item(),
           'AUC': auc,
           'Bias': bias}

class Evaluator(_BaseEvaluator):
  def __init__(self, test_dataloader, num_bias, cpu_eval=False, take_compo_scores=True,  open_world= False):
    super().__init__(test_dataloader, num_bias, cpu_eval)
    self.take_compo_scores = take_compo_scores
    self.open_world= open_world
    
  def get_composcores(self, attr_scores, obj_scores):
    obj_preds = torch.softmax(obj_scores, dim=-1)
    attr_preds = torch.softmax(attr_scores, dim=-1)
    return torch.bmm(attr_preds.unsqueeze(2), obj_preds.unsqueeze(1))
  
  def get_primitive_preds(self, compo_scores, topk):
    ncol = compo_scores.shape[-1]
    _, topk_preds = torch.topk(compo_scores.view(len(compo_scores), -1), topk, dim=-1) # [batch, k]
    topk_attr_preds, topk_obj_preds = self.pairId2primitiveId(topk_preds)
    return topk_attr_preds, topk_obj_preds    
  
  def eval_primitive_scores(self, attr_scores, obj_scores, topk=1):
    """Return: Tuple of (closed_world_report, open_word_report).
    report: best_seen, best_unseen, best_harmonic, auc
    """
    compo_scores = self.get_composcores(attr_scores, obj_scores)
    _, attr_preds = torch.topk(attr_scores, topk, axis=-1)
    _, obj_preds = torch.topk(obj_scores, topk, axis=-1)
    return self.evaluate(attr_preds, obj_preds, compo_scores, topk)

  def eval_compo_scores(self, compo_scores, topk=1):
    
    acc, acc_biased, biaslist = self.compo_acc(compo_scores, topk)
    breport = self.analyse_acc_report(acc_biased, biaslist)
    if not self.test_dataloader.dataset.name == 'CGQA':
      compo_scores[:, ~self.seen_mask] += 0.95*breport['Bias'] 
    ncol = compo_scores.shape[-1]
    _, topk_preds = torch.topk(compo_scores.view(len(compo_scores), -1), topk, dim=-1) # [batch, k]
    topk_attr_preds, topk_obj_preds = self.pairId2primitiveId(topk_preds)
    attr_acc = self.acc(topk_attr_preds, self.attr_labels)
    obj_acc = self.acc(topk_obj_preds, self.obj_labels)
  
    report = {'A': attr_acc, 'O': obj_acc}
    report.update({k:v for k, v in breport.items()})
    
    self.attr_labels, self.obj_labels = None, None # drop labels of old batch
    return report
  
  def eval_output(self, output, attr_labels, obj_labels, ow_fscore = None, topk=1, no_bias=False):
    self.attr_labels = attr_labels.to(self.dev)
    self.obj_labels = obj_labels.to(self.dev)
    self.no_bias = no_bias

    if self.open_world:
      fscore_threshold = 0.2
      self.fscore_mask = ow_fscore < fscore_threshold
      
    if self.take_compo_scores:
      if isinstance(output, list):
        if isinstance(output[0], tuple):
          output = list(zip(*output))[0]
        compo_scores = torch.cat(output)
      else:
        compo_scores = output
      compo_scores = compo_scores.to(self.dev).reshape(-1, self.attr_class, self.obj_class)
      return self.eval_compo_scores(compo_scores.to(self.dev), topk=topk)
    else:
      if isinstance(output, list):
        attr_scores, obj_scores = list(zip(*output))[:2]
        attr_scores = torch.cat(attr_scores)
        obj_scores = torch.cat(obj_scores)
      else:
        attr_scores, obj_scores = output
      return self.eval_primitive_scores(attr_scores.to(self.dev), obj_scores.to(self.dev), topk=topk)

class _IREvaluator:
  def __init__(self, cpu_eval):
    self.dev = 'cpu' if cpu_eval else dev
    
  def eval_output(self, output, attr_labels, obj_labels):
    pass

class IREvaluator(_IREvaluator):
  def recall(self, preds, labels):
    # preds: [nsample, k]
    nsample = len(preds)
    labels = labels.unsqueeze(1)
    rec = (preds == labels).any(dim=1).float().mean()
    return rec
  
  def eval_output(self, output, attr_labels, obj_labels):
    output = list(zip(*output))
    theta = torch.cat(output[0])
    t_pair_ids = torch.cat(output[1])
    img_feats = torch.cat(output[2])
    pair_ids = torch.cat(output[3]).view(-1)

    dot = theta @ img_feats.T
    mask = torch.zeros_like(dot, dtype=torch.bool)
    for i in range(min(len(theta), len(img_feats))):
      mask[i,i] = True
    dot[mask] = -1e5
    report = dict()
    for k in [1, 10, 50]:
      predictions = dot.topk(k, dim=1).indices # nqueries * topk
      pred_pairs = pair_ids[predictions]
      rec = self.recall(pred_pairs, t_pair_ids)
      report[f'IR_Rec/top{k}'] = rec.item()
    return report
  
class Fashion200kWrapper:
  def __init__(self, dset):
    self.dset = dset
    
  def __len__(self):
    return len(self.dset.test_imgs)
  
  def __getitem__(self, idx):
    img_feat = self.dset.get_img(idx)
    img = self.dset.test_imgs[idx]
    return [img_feat, img['captions'][0], self.dset.obj2idx[img['obj']]]
  
class IREvaluatorFashion(_IREvaluator):
  def __init__(self, cpu_eval, dset, model):
    self.dev = 'cpu' if cpu_eval else dev
    self.dset = dset
    self.model = model
    
  def extract_dataset(self, model):
    from torch.utils.data import DataLoader
    model.eval()
    batch_size=128
    loader = DataLoader(Fashion200kWrapper(self.dset), batch_size, False)
    img_feats = []
    captions = []
    objs = []
    print("Extracting test dataset for evaluation.")
    with torch.no_grad():
      for batch in tqdm.tqdm(loader):
        img = batch[0]
        if model.resnet:
          img = model.resnet(img.to(dev))
        img_feats.append(model.img_fc(img))
        captions += batch[1]
        objs.append(batch[2])
    img_feats = torch.cat(img_feats).to(self.dev)
    objs = torch.cat(objs).to(self.dev)
    return img_feats, captions, objs
    
  def recall(self, t_attr_id, obj_id, caption_preds, obj_preds):
    obj_match = (obj_id.unsqueeze(1) == obj_preds).any(dim=1)
    t_attr = [self.dset.attrs[i] for i in t_attr_id]
    attr_match = []
    for i in range(len(t_attr_id)):
      attr_match.append(any([t_attr[i] in caption for caption in caption_preds[i]]))
    attr_match = torch.tensor(attr_match, dtype=torch.bool).to(self.dev)
    rec = (obj_match & attr_match).float().mean()
    return rec
  
  def recall_caption(self, caption_preds, captions):
    match = []
    for pred, caption in zip(caption_preds, captions):
      match.append(any([caption==caption_pred for caption_pred in pred]))
    return torch.tensor(match).float().mean()
  
  def eval_output(self, output, attr_id, obj_id):
    self.img_feats, self.captions, self.obj_ids = self.extract_dataset(self.model)
    t_caption = []
    for batch in output:
      t_caption += batch[-1]
    output = list(zip(*output))

    theta = torch.cat(output[0]).to(self.dev)
    t_attr_id = torch.cat(output[1]).to(self.dev)
    s_img_idx = torch.cat(output[2]).to(self.dev)
    t_obj_id = obj_id.to(self.dev)

    dot = theta @ self.img_feats.T
    dot[range(len(dot)), s_img_idx] = -1e10
    
    report = dict()
    for k in [1, 10, 50]:
      predictions = dot.topk(k, dim=1).indices # nqueries * topk
      caption_preds = []
      for k_pred in predictions:
        captions = [self.captions[i] for i in k_pred]
        caption_preds.append(captions)
      rec = self.recall_caption(caption_preds, t_caption)
#       obj_preds = self.obj_ids[predictions]
#       rec = self.recall(t_attr_id, t_obj_id, caption_preds, obj_preds)
      report[f'IR_Rec/top{k}'] = rec.item()
    return report