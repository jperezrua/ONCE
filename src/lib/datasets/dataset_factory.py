from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.epdet import EpisodicDetDataset
from .sample.mixdet import MixDetDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.coco_base import COCOBase
from .dataset.coco_episodic import COCOEpisodic
from .dataset.coco_mixed import COCOMixed

dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'coco_base': COCOBase,
  'coco_ep': COCOEpisodic,
  'coco_mixed': COCOMixed
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset,
  'epdet': EpisodicDetDataset,
  'mixdet': MixDetDataset
}


def get_dataset(dataset, task):
  print('Creating Dataset: {}, {} ...'.format(dataset, task))
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
