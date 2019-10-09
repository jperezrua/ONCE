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

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.query_images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.support_images = self._get_support_images(dataset)

  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_support_images(self, dataset):
    support_per_cat = {cat:dataset.coco_support.getImgIds(catIds=cat) for cat in dataset._valid_ids}

    support_images = []
    for cat in support_per_cat:
      img_ids    = np.random.choice(dataset.coco_support.getImgIds(catIds=cat), self.opt.k_shots).tolist()    
      img_items  = dataset.coco_support.loadImgs(ids=img_ids)

      img_paths = []
      supp_anns = []
      ims = []
      # this for loop is to take one randomly sampled annotation (give is cat_id) for each of the k_shots
      for img_id, img_i in zip(img_ids, img_items):
        img_paths.append(os.path.join(dataset.supp_img_dir, img_i['file_name']))
        ann_ids    = dataset.coco_support.getAnnIds(imgIds=[img_id])
        anns       = dataset.coco_support.loadAnns(ids=ann_ids)
        valid_anns = [a for a in anns if a['category_id']==cat]
        supp_anns.append(np.random.choice(valid_anns))

      supp_imgs = [cv2.imread(img_path) for img_path in img_paths]
      
      out_supp = []
      for img, ann in zip(supp_imgs, supp_anns):
        bbox = self._coco_box_to_bbox(ann['bbox'])
        x1,y1,x2,y2 = math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3])
        inp = img[y1:y2,x1:x2,:]

        inp = cv2.resize(inp, (int(self.opt.supp_w), int(self.opt.supp_h)))
        ims.append(inp)
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
    print(support_images.shape, device)
    return support_images

  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
      if opt.task == 'ddd':
        images[scale], meta[scale] = self.pre_process_func(
          image, scale, img_info['calib'])
      else:
        images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.images)

def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split, base=False)
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
  dataset.run_eval(results, opt.save_dir)

def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])

    if opt.task == 'ddd':
      ret = detector.run(img_path, img_info['calib'])
    else:
      ret = detector.run(img_path)
    
    results[img_id] = ret['results']

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()
  dataset.run_eval(results, opt.save_dir)

if __name__ == '__main__':
  opt = opts().parse()
  if opt.not_prefetch_test:
    test(opt)
  else:
    prefetch_test(opt)