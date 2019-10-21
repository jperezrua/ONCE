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

  def _sample_query_from_categories(self, sampled_categories):


    # this loop is to sample a single image for every category 
    # (to be sure each cat gets at least an image)
    query_img_paths = []
    anns_per_query  = []
    category_dict   = {}
    for idx,cat in enumerate(sampled_categories):
      image_ids = self.coco.getImgIds(catIds=cat)
      img_id    = random.choice(image_ids)
      file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
      img_path  = os.path.join(self.img_dir, file_name)
      ann_ids   = self.coco.getAnnIds(imgIds=[img_id])
      all_anns  = self.coco.loadAnns(ids=ann_ids)
      val_anns  = [a for a in all_anns if a['category_id'] in sampled_categories]

      anns_per_query.append( val_anns )
      query_img_paths.append( img_path )
      category_dict[cat] = idx

    return query_img_paths, anns_per_query, category_dict

  def _sample_support_set(self, cat_id):

    img_ids    = self.coco_support.getImgIds(catIds=[cat_id])
    #img_items  = self.coco_support.loadImgs(ids=img_ids)
    ann_ids    = self.coco_support.getAnnIds(imgIds=img_ids)
    anns       = self.coco_support.loadAnns(ids=ann_ids)

    is_proper_size = lambda a: (a['bbox'][2]>=self.opt.min_bbox_len) & (a['bbox'][3]>=self.opt.min_bbox_len)
    is_proper_cat = lambda a:a['category_id']==cat_id
    good_anns = [a for a in anns if (is_proper_size(a) & is_proper_cat(a))]
    sampled_good_anns = np.random.choice(good_anns, self.k_shots).tolist()

    img_paths = []    
    for s in sampled_good_anns:
      img_file_name = self.coco_support.loadImgs([s['image_id']])[0]['file_name']
      img_paths.append(os.path.join(self.supp_img_dir, img_file_name))

    return img_paths, sampled_good_anns

  def _process_query(self, img, cat, augment=False):
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

    #cv2.imshow('inp-{}'.format(cat),inp)

    inp = (inp.astype(np.float32) / 255.)
    if augment and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)
    inp_dim = (input_h, input_w)
    return inp, inp_dim, flipped, center, scale

  def _process_all_query_outs(self, query_imgs, anns_per_query, query_info, category_dict):

    hm_per_query = []
    reg_mask_per_query = []
    reg_per_query = []
    ind_per_query = []
    wh_per_query = []
    cs_wh_per_query = []
    cs_mask_per_query = []
    gt_det_per_query = []

    for query_idx, img in enumerate(query_imgs):
      width = img.shape[2]#(2, 0, 1)
      input_h, input_w = query_info['inp_dim'][query_idx]
      output_h = input_h // self.opt.down_ratio
      output_w = input_w // self.opt.down_ratio
      num_classes = len(query_info['sampled_categories'])
      center = query_info['center'][query_idx]
      scale = query_info['scale'][query_idx]
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
      num_objs = min(len(anns_per_query[query_idx]), self.max_objs)
      for k in range(num_objs):
        ann = anns_per_query[query_idx][k]
        bbox = self._coco_box_to_bbox(ann['bbox'])
        cls_id = category_dict[ann['category_id']]
        if query_info['flipped'][query_idx]:
          bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        bbox[:2] = affine_transform(bbox[:2], trans_output)
        bbox[2:] = affine_transform(bbox[2:], trans_output)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h > 0 and w > 0:
          radius = gaussian_radius((math.ceil(h), math.ceil(w)))
          radius = max(0, int(radius))
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

      #cv2.imshow( 'hm-query-{}-cat-{}'.format(query_idx,0), cv2.resize(hm[0], tuple(img.shape[1:3])) )
      #cv2.imshow( 'hm-query-{}-cat-{}'.format(query_idx,1), cv2.resize(hm[1], tuple(img.shape[1:3])) )
      #cv2.imshow( 'hm-query-{}-cat-{}'.format(query_idx,2), cv2.resize(hm[2], tuple(img.shape[1:3])) )
      hm_per_query.append(hm)
      reg_mask_per_query.append(reg_mask)
      reg_per_query.append(reg)
      ind_per_query.append(ind)
      wh_per_query.append(wh)
      gt_det_per_query.append(gt_det)
      cs_wh_per_query.append(cat_spec_wh)
      cs_mask_per_query.append(cat_spec_mask)

    hm = np.stack(hm_per_query)
    reg_mask = np.stack(reg_mask_per_query)
    reg = np.stack(reg_per_query)
    ind = np.stack(ind_per_query)
    wh  = np.stack(wh_per_query)
    cs_wh_per_query = np.stack(cs_wh_per_query)
    cs_mask_per_query = np.stack(cs_mask_per_query)

    return hm, reg_mask, reg, ind, wh, gt_det_per_query, cs_wh_per_query, cs_mask_per_query

  def _process_support_set(self, support_imgs, support_anns, cat, augment=False):

    out_supp = []
    for i, (img, ann) in enumerate(zip(support_imgs, support_anns)):
      bbox = self._coco_box_to_bbox(ann['bbox'])
      x1,y1,x2,y2 = math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3])
      
      #give a little more of context for support
      y1 = max(0, y1-self.opt.supp_ctxt)
      x1 = max(0, x1-self.opt.supp_ctxt)
      y2 = min(y2+self.opt.supp_ctxt, img.shape[0])
      x2 = min(x2+self.opt.supp_ctxt, img.shape[1])

      inp = img[y1:y2,x1:x2,:]

      if augment:
        if np.random.random() < self.opt.flip:
          inp = inp[:, ::-1, :]

      #cv2.imshow('sample-{}-cat-{}'.format(i,cat), inp)

      inp = cv2.resize(inp, (int(self.opt.supp_w), int(self.opt.supp_h)))
      inp = (inp.astype(np.float32) / 255.)
      inp = (inp - self.mean) / self.std
      inp = inp.transpose(2, 0, 1)
      out_supp.append(inp)
    out_supp = np.stack(out_supp,axis=0)

    return out_supp

  def _sample_categories(self,num_categories):
    cat_ids = random.sample(self._valid_ids, num_categories)
    return cat_ids


  def __getitem__(self, index):

    # 1. sample n categories
    sampled_categories = self._sample_categories(self.n_sample_classes)

    # 2. sample one image per category and load annotations for each image
    query_img_paths, anns_per_query, category_dict = self._sample_query_from_categories(sampled_categories)
    
    # 3. load all the query images and process them
    query_imgs = []
    query_info = {'flipped': [], 'center': [], 'scale': [], 'inp_dim': [], 'sampled_categories': sampled_categories}
    for qi,path in enumerate(query_img_paths):
      query_img = cv2.imread(path)
      inp, inp_dim, flipped, center, scale = self._process_query(query_img, qi, augment=(self.split=='train'))
      query_imgs.append(inp)
      query_info['flipped'].append(flipped)
      query_info['center'].append(center)
      query_info['scale'].append(scale)
      query_info['inp_dim'].append(inp_dim)

    # 4. sample and process the support set
    support_set = []
    for ic,cat in enumerate(sampled_categories):
      support_paths, support_anns = self._sample_support_set(cat)
      supp_imgs = [cv2.imread(img_path) for img_path in support_paths]
      supp_imgs = self._process_support_set(supp_imgs, support_anns, ic, augment=(self.split=='train'))
      support_set.append(supp_imgs)
    support_set = np.stack(support_set,axis=0)

    # 5. Process query gt output
    hm, reg_mask, reg, ind, wh, gt_det, cs_wh_per_query, cs_mask_per_query = self._process_all_query_outs(query_imgs, anns_per_query, query_info, category_dict)
    

    # 6. stack all together to be size [N,...]
    query_imgs = np.stack(query_imgs, axis=0)
    
    #cv2.waitKey(0)
    #print(query_imgs.shape, hm.shape, wh.shape, support_set.shape,'**************')

    ret = {'input': query_imgs, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 
           'supp': support_set, 'cat_spec_wh': cs_wh_per_query, 'cat_spec_mask': cs_mask_per_query}

    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det[0], dtype=np.float32) if len(gt_det[0]) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      #meta = {'c': center, 's': scale, 'gt_det': gt_det, 'img_id': query_id}
      meta = {'c': center, 's': scale, 'gt_det': gt_det}
      ret['meta'] = meta
    return ret
