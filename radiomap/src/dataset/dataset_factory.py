from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .datasets.SimRsrp import SimRsrpDataset
from .datasets.RealRsrp import RealRsrpDataset
from .datasets.RadioMap3D import RadioMap3D


dataset_factory = {
  'real': RealRsrpDataset,
  'simulation': SimRsrpDataset,
  'radiomap3d': RadioMap3D,
}


def get_dataset(dataset):
  return dataset_factory[dataset]

