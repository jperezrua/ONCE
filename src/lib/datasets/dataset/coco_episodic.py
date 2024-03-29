from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class COCOEpisodic(data.Dataset):
  default_resolution = [512, 512]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  num_classes = 1

  def __init__(self, opt, split, base=True):
    super(COCOEpisodic, self).__init__()

    self.base = base
    self.k_shots = opt.k_shots
    if base:
      self.num_classes = 60
    else:
      self.num_classes = 20

    if split == 'train':
      self.n_sample_classes = opt.n_class
    else:
      self.n_sample_classes = 1

    if self.base:
      assert not opt.keep_res

    print('COCOEpisodic with {} classes'.format(self.num_classes))
    

    self.data_dir = os.path.join(opt.data_dir, 'coco')
    self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
    print(100*'*')
    print(self.img_dir, split)

    self.is_train = base

    if opt.task == 'epdet':
      if base:
        # if 'base' we will use the instances_base_train2017.json file for
        # sampling both the query and support sets				
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'instances_base_{}2017.json'.format(split))
        self.annot_supp_path = os.path.join(
          self.data_dir, 'annotations', 
          'instances_base_{}2017.json'.format(split))
        self.supp_img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
      else:
        # if 'novel' we will use the instances_novel_train2017.json file for
        # sampling the support set, and instances_novel_val2017.json for query
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'instances_novel_val2017.json')
        self.annot_supp_path = os.path.join(
          self.data_dir, 'annotations', 
          'instances_novel_train2017.json')
        self.supp_img_dir = os.path.join(self.data_dir, 'train2017')
              
    self.max_objs = 128
    
    val_categories = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "boat",
            "bird", "cat", "dog", "horse", "sheep", "cow", "bottle", "chair", "couch", 
            "potted plant", "dining table", "tv"
                     ]    
                       
    self.class_name = [
      '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
      'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
      'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
      'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
      'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
      'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
      'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
      'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
      'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
      'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
      'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
      'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    train_categories = [c for c in self.class_name if c != '__background__' and c not in val_categories]

    all_ids = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
      24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
      37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
      58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
      72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
      82, 84, 85, 86, 87, 88, 89, 90]

    train_ids = [
      8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
      39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
      61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    val_ids = [id for id in all_ids if id not in train_ids]

    if base:
      self._valid_ids = train_ids
      for nac in val_categories:
        self.class_name.remove(nac)
    else:
      self._valid_ids = val_ids
      for nac in train_categories:
        self.class_name.remove(nac)        
      
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.split = split
    self.opt = opt

    print('==> initializing coco 2017 {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.coco_support = coco.COCO(self.annot_supp_path)
    
    self.query_images = self.coco.getImgIds()
    #self.images_per_cat = {cat:self.coco_support.getImgIds(catIds=cat) for cat in self._valid_ids}

    self.num_samples = len(self.query_images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    if not self.is_train:
      return self.num_samples
    else:
      return min(self.num_samples, 10000) # randomly sampled episodes

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
  
  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)

    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats
