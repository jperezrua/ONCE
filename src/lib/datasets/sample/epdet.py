from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math
import random

class EpisodicDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def _sample_query(self, index):
    query_id  = self.query_images[index]
    file_name = self.coco.loadImgs(ids=[query_id])[0]['file_name']
    img_path  = os.path.join(self.img_dir, file_name)
    ann_ids   = self.coco.getAnnIds(imgIds=[query_id])
    anns      = self.coco.loadAnns(ids=ann_ids)
    cats_anns = np.unique([a['category_id'] for a in anns])
    query_cat = np.random.choice(cats_anns)
    query_anns= [a for a in anns if a['category_id']==query_cat]
    return img_path, query_anns, query_cat, query_id

  def _sample_support_set(self, cat_id):
    img_ids    = np.random.choice(self.coco_support.getImgIds(catIds=[cat_id]), self.k_shots).tolist()
    img_items  = self.coco_support.loadImgs(ids=img_ids)

    img_paths = []
    supp_anns = []

    # this for loop is to take one randomly sampled annotation (give is cat_id) for each of the k_shots
    for img_id, img_i in zip(img_ids, img_items):
      img_paths.append(os.path.join(self.supp_img_dir, img_i['file_name']))
      ann_ids    = self.coco_support.getAnnIds(imgIds=[img_id])
      anns       = self.coco_support.loadAnns(ids=ann_ids)
      valid_anns = [a for a in anns if a['category_id']==cat_id]
      supp_anns.append(np.random.choice(valid_anns))

    return img_paths, supp_anns

  def _process_query(self, img, augment=False):
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      scale = np.array([input_w, input_h], dtype=np.float32)
    else:
      scale = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w
    
    flipped = False
    if augment:
      if not self.opt.not_rand_crop:
        scale = scale * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        center[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        center[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        center[0] += scale * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        center[1] += scale * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        scale = scale * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        center[0] =  width - center[0] - 1
        
    trans_input = get_affine_transform(
      center, scale, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if augment and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)
    inp_dim = (input_h, input_w)
    return inp, inp_dim, flipped, center, scale

  def _process_query_out(self, img, anns, flipped, center, scale, input_dim, num_objs):
    width = img.shape[1]
    input_h, input_w = input_dim
    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = 1 #TODO: FIND A BETTER FIX
    trans_output = get_affine_transform(center, scale, 0, [output_w, output_h])

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      cls_id = 0#TODO: BETTER FIX #int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = radius+0.1#self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
    return hm, reg_mask, reg, ind, wh, gt_det

  def _process_support_set(self, support_imgs, support_anns, augment=False):

    out_supp = []
    for img, ann in zip(support_imgs, support_anns):
      bbox = self._coco_box_to_bbox(ann['bbox'])
      x1,y1,x2,y2 = math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3])
      inp = img[y1:y2,x1:x2,:]

      if augment:
        if np.random.random() < self.opt.flip:
          inp = inp[:, ::-1, :]

      inp = cv2.resize(inp, (int(self.opt.supp_w), int(self.opt.supp_h)))
      inp = (inp.astype(np.float32) / 255.)
      inp = (inp - self.mean) / self.std
      inp = inp.transpose(2, 0, 1)
      out_supp.append(inp)
    out_supp = np.stack(out_supp,axis=0)
    return out_supp

  def __getitem__(self, index):

    # 1. Extract the query image and get annotation for a single category
    query_path, query_anns, query_cat, query_id = self._sample_query(index)
    query_img = cv2.imread(query_path)
    num_objs  = min(len(query_anns), self.max_objs)

    # 2. Now take the query category and sample k_shots support samples
    support_paths, support_anns = self._sample_support_set(query_cat)
    supp_imgs = [cv2.imread(img_path) for img_path in support_paths]

    # 3. Process query image and get scale and center
    inp, inp_dim, flipped, center, scale = self._process_query(query_img, augment=(self.split=='train'))

    # 4. Process query gt output
    hm, reg_mask, reg, ind, wh, gt_det = self._process_query_out(query_img, query_anns, 
                                          flipped, center, scale, inp_dim, num_objs)
    
    # 5. Process support imgs
    supp_imgs = self._process_support_set(supp_imgs, support_anns, augment=(self.split=='train'))
    
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'supp': supp_imgs}

    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': center, 's': scale, 'gt_det': gt_det, 'img_id': query_id}
      ret['meta'] = meta
    return ret
