from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
import math

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
import random

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.coco.getImgIds()
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

    if opt.task != 'reweight_paper':
      self.support_images = self._get_support_images(dataset)
    else:
      self.support_images = self._get_support_images_rw(dataset)

  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_support_images_rw(self, dataset):
    support_set = []
    for catid in dataset._fewshot_ids:
      img_ids    = dataset.coco_supp.getImgIds(catIds=catid)
      ann_ids    = dataset.coco_supp.getAnnIds(imgIds=img_ids)
      good_anns  = dataset.coco_supp.loadAnns(ids=ann_ids) #good_anns[:self.opt.k_shots]

      good_anns = [a for a in good_anns if a['category_id'] == catid]

      if len(good_anns)>=self.opt.k_shots:
        sampled_good_anns = random.sample(good_anns, self.opt.k_shots)
      else:
        sampled_good_anns = [random.choice(good_anns) for _ in range(self.opt.k_shots)]

      supp_for_catid = []

      for sampleid, ann in enumerate(sampled_good_anns):
        #print(catid, ann)
        img_file_name = dataset.coco_supp.loadImgs([ann['image_id']])[0]['file_name']
        img_path = os.path.join(dataset.supp_img_dir, img_file_name)
        img = cv2.imread(img_path)

        bbox = self._coco_box_to_bbox(ann['bbox'])
        x1,y1,x2,y2 = math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3])
        
        #give a little more of context for support
        y1 = max(0, y1-self.opt.supp_ctxt)
        x1 = max(0, x1-self.opt.supp_ctxt)
        y2 = min(y2+self.opt.supp_ctxt, img.shape[0])
        x2 = min(x2+self.opt.supp_ctxt, img.shape[1])

        inp = img[y1:y2,x1:x2,:]
        #cv2.imshow('cat-{}-sample-{}'.format(catid,sampleid), inp)

        inp = cv2.resize(inp, (int(self.opt.supp_w), int(self.opt.supp_h)))
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        supp_for_catid.append(inp)
      supp_for_catid = np.stack(supp_for_catid,axis=0)
      support_set.append(supp_for_catid)
      #print('********* ',catid, supp_for_catid.shape)

    support_images = np.stack(support_set,axis=0)
    support_images = torch.from_numpy(support_images)
    device = torch.device('cuda') if self.opt.gpus[0] >= 0 else torch.device('cpu')  
    support_images = support_images.to(device)
    return support_images

  def _get_support_images(self, dataset):
    #support_per_cat = {cat:dataset.coco_support.getImgIds(catIds=cat) for cat in dataset._valid_ids}
    support_per_cat = [(cat,dataset.coco_support.getImgIds(catIds=cat)) for cat in dataset._valid_ids]

    #print('All categories: ',dataset._valid_ids)
    #print('All categories: ',dataset._pascal_valid_ids)
    

    support_images = []
    cats = []
    for elem in support_per_cat:
      cat = elem[0]
      cats.append(cat)
      #img_ids    = np.random.choice(elem[1], self.opt.k_shots).tolist()    
      #img_items  = dataset.coco_support.loadImgs(ids=img_ids)
      img_ids    =  elem[1]
      ann_ids    = dataset.coco_support.getAnnIds(imgIds=img_ids)
      anns       = dataset.coco_support.loadAnns(ids=ann_ids)
      #print('Cat: {} + Len anns: {} + Len img ids: {}'.format(cat,len(anns),len(img_ids)))
      is_proper_size = lambda a: (a['bbox'][2]>=self.opt.min_bbox_len) & (a['bbox'][3]>=self.opt.min_bbox_len)
      is_proper_cat = lambda a:a['category_id']==cat
      good_anns = [a for a in anns if (is_proper_size(a) & is_proper_cat(a))]
      sampled_good_anns = np.random.choice(good_anns, dataset.k_shots).tolist()

      img_paths = []
      for s in sampled_good_anns:
        img_file_name = dataset.coco_support.loadImgs([s['image_id']])[0]['file_name']
        img_paths.append(os.path.join(dataset.supp_img_dir, img_file_name))

      supp_imgs = [cv2.imread(img_path) for img_path in img_paths]
      
      out_supp = []
      for i, (img, ann) in enumerate(zip(supp_imgs, sampled_good_anns)):
        bbox = self._coco_box_to_bbox(ann['bbox'])
        x1,y1,x2,y2 = math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3])

        y1 = max(0, y1-self.opt.supp_ctxt)
        x1 = max(0, x1-self.opt.supp_ctxt)
        y2 = min(y2+self.opt.supp_ctxt, img.shape[0])
        x2 = min(x2+self.opt.supp_ctxt, img.shape[1])

        inp = img[y1:y2,x1:x2,:]
        if self.opt.debug == 5:
          cv2.imshow('supp_{}_cat{}'.format(i,cat),inp)

        inp = cv2.resize(inp, (int(self.opt.supp_w), int(self.opt.supp_h)))
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        out_supp.append(inp)

      #for k,im in enumerate(ims):
      #  cv2.imshow('supp_{}_cat{}'.format(k,cat),im)
      #cv2.waitKey(0)

      out_supp = np.stack(out_supp,axis=0)
      support_images.append(out_supp)
    support_images = np.stack(support_images,axis=0)
    support_images = torch.from_numpy(support_images)
    device = torch.device('cuda') if self.opt.gpus[0] >= 0 else torch.device('cpu')  
    support_images = support_images.to(device)
    #print(support_images.shape, device)
    #print(dataset._valid_ids,cats)
    return support_images

  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    #print('Test image location: ', img_path)
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
      images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.images)

def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  if opt.task != 'reweight_paper':
    Dataset = dataset_factory[opt.dataset]
  else:
    Dataset = dataset_factory['coco_mixed']

  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
    
  split = 'val'# if not opt.trainval else 'test'

  if opt.task != 'reweight_paper':
    dataset = Dataset(opt, split, base=False)
  else:
    dataset = Dataset(opt, split)

  opt.num_classes = dataset.num_classes
  
  if opt.task != 'reweight_paper':
    Detector = detector_factory[opt.task]
  else:
    Detector = detector_factory['epdet']
  detector = Detector(opt)
  
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  #this computes the support codes for a batch of supports with size [C,B=1,K_shots,img_dims...]

  y_codes = detector.model.precompute_multi_class(data_loader.dataset.support_images) 

  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    ret = detector.run(pre_processed_images, y_codes)
    results[img_id.numpy().astype(np.int32)[0]] = ret['results']
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
  bar.finish()
  stats = dataset.run_eval(results, opt.save_dir)
  return stats

if __name__ == '__main__':
  opt = opts().parse()
  list_stats = []
  for i in range(100):
    print('************** iteration {} ***************'.format(i))
    list_stats.append( np.array( prefetch_test(opt) ) )
    if i>0:
      stats_ = np.stack(list_stats,axis=0)  
      print(np.mean(stats_,axis=0))
      print(np.std(stats_,axis=0))   
  stats = np.stack(list_stats,axis=0)
  
  print('STATS')
  print(np.mean(stats,axis=0))
  print(np.std(stats,axis=0))