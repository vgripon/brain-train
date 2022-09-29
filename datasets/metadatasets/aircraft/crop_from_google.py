from PIL import Image
from PIL import ImageOps
from absl import logging
import io
import os
import itertools
import sys
import json
import collections

sys.path.insert(0,'./../../../')
from utils import *


class AircraftConverter():
  """Prepares Aircraft as required to integrate it in the benchmark."""
  # There are 100 classes in the Aircraft dataset. A 70% / 15% / 15%
  # split between train, validation and test maps to 70 / 15 / 15
  # classes, respectively.
  def __init__(self):
    self.NUM_TRAIN_CLASSES = 70
    self.NUM_VALID_CLASSES = 15
    self.NUM_TEST_CLASSES = 15
    self.data_root = os.path.join(args.dataset_path, 'fgvc-aircraft-2013b')
    self.classes_per_split = {
        0: 0,
        1: 0,
        2: 0
    }
    self.class_names = {}
    self.images_per_class = collections.defaultdict(int)

  def split_fn(self,json_path = 'aircraft_splits.json'):
    with open(json_path) as jsonFile:
        split = json.load(jsonFile)
        jsonFile.close()
    return split



  def get_splits(self):
    """Create splits for Aircraft and store them in the default path.
    If no split file is provided, and the default location for Aircraft splits
    does not contain a split file, splits are randomly created in this
    function using 70%, 15%, and 15% of the data for training, validation and
    testing, respectively, and then stored in that default location.
    Returns:
      The splits for this dataset, represented as a dictionary mapping each of
      'train', 'valid', and 'test' to a list of strings (class names).
    """
    
    # "Variant" refers to the aircraft model variant (e.g., A330-200) and is
    # used as the class name in the dataset.
    variants_path = os.path.join(self.data_root, 'data', 'variants.txt')
    with open(variants_path, 'r') as f:
      variants = [line.strip() for line in f.readlines() if line]
    variants = sorted(variants)
    assert len(variants) == (
        self.NUM_TRAIN_CLASSES + self.NUM_VALID_CLASSES + self.NUM_TEST_CLASSES)

    splits = self.split_fn()
    return splits

  def create_dataset_specification_and_records(self):
    """Implements DatasetConverter.create_dataset_specification_and_records."""

    splits = self.get_splits()
    # Get the names of the classes assigned to each split
    train_classes = splits['train']
    valid_classes = splits['valid']
    test_classes = splits['test']

    self.classes_per_split[0] = len(train_classes)
    self.classes_per_split[1] = len(valid_classes)
    self.classes_per_split[2] = len(test_classes)

    # Retrieve mapping from filename to bounding box.
    # Cropping to the bounding boxes is important for two reasons:
    # 1) The dataset documentation mentions that "[the] (main) aircraft in each
    #    image is annotated with a tight bounding box [...]", which suggests
    #    that there may be more than one aircraft in some images. Cropping to
    #    the bounding boxes removes ambiguity as to which airplane the label
    #    refers to.
    # 2) Raw images have a 20-pixel border at the bottom with copyright
    #    information which needs to be removed. Cropping to the bounding boxes
    #    has the side-effect that it removes the border.
    bboxes_path = os.path.join(self.data_root, 'data', 'images_box.txt')
    with open(bboxes_path, 'r') as f:
      names_to_bboxes = [
          line.split('\n')[0].split(' ') for line in f.readlines()
      ]
      names_to_bboxes = dict(
          (name, list(map(int, (xmin, ymin, xmax, ymax))))
          for name, xmin, ymin, xmax, ymax in names_to_bboxes)

    # Retrieve mapping from filename to variant
    variant_trainval_path = os.path.join(self.data_root, 'data',
                                         'images_variant_trainval.txt')
    with open(variant_trainval_path, 'r') as f:
      names_to_variants = [
          line.split('\n')[0].split(' ', 1) for line in f.readlines()
      ]

    variant_test_path = os.path.join(self.data_root, 'data',
                                     'images_variant_test.txt')
    with open(variant_test_path, 'r') as f:
      names_to_variants += [
          line.split('\n')[0].split(' ', 1) for line in f.readlines()
      ]

    names_to_variants = dict(names_to_variants)

    # Build mapping from variant to filenames. "Variant" refers to the aircraft
    # model variant (e.g., A330-200) and is used as the class name in the
    # dataset. The position of the class name in the concatenated list of
    # training, validation, and test class name constitutes its class ID.
    variants_to_names = collections.defaultdict(list)
    for name, variant in names_to_variants.items():
      variants_to_names[variant].append(name)

    all_classes = list(
        itertools.chain(train_classes, valid_classes, test_classes))
    assert set(variants_to_names.keys()) == set(all_classes)

    for class_id, class_name in enumerate(all_classes):
      logging.info('Creating record for class ID %d (%s)...', class_id,
                   class_name)
      class_files = [
          os.path.join(self.data_root, 'data', 'images',
                       '{}.jpg'.format(filename))
          for filename in sorted(variants_to_names[class_name])
      ]

      bboxes = [
          names_to_bboxes[name]
          for name in sorted(variants_to_names[class_name])
      ]

      self.class_names[class_id] = class_name
      self.images_per_class[class_id] = len(class_files)
      write_from_image_files(class_files, class_id, bboxes=bboxes, new_path = 'fgvc-aircraft-2013b/data/images_cropped2/')




if __name__ == "__main__":
    conv = AircraftConverter()
    conv.get_splits()
    conv.create_dataset_specification_and_records()
