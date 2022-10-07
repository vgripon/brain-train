from PIL import Image
from PIL import ImageOps
from absl import logging
import io
import os
import itertools
import sys
import json
import collections
import numpy as np
sys.path.insert(0,'./../../../')
from utils import *
from args import args
from tqdm import tqdm

class MSCOCOConverter():
  """Prepares MSCOCO as required to integrate it in the benchmark."""

  # There are 80 classes in the MSCOCO dataset. A 0% / 50% / 50% split
  # between train, validation and test maps to roughly 0 / 40 / 40 classes,
  # respectively.
  NUM_TRAIN_CLASSES = 0
  NUM_VALID_CLASSES = 40
  NUM_TEST_CLASSES = 40

  def __init__(self,
               data_root,
               image_subdir_name='train2017',
               annotation_json_name='instances_train2017.json',
               box_scale_ratio=1.2):
    self.box_scale_ratio=box_scale_ratio
    self.num_all_classes = (
        self.NUM_TRAIN_CLASSES + self.NUM_VALID_CLASSES + self.NUM_TEST_CLASSES)
    image_dir = os.path.join(data_root, image_subdir_name)
    if not os.path.isdir(image_dir):
      raise ValueError('Directory %s does not exist' % image_dir)
    self.image_dir = image_dir
    self.data_root = data_root
    annotation_path = os.path.join(data_root, annotation_json_name)
    if not os.path.exists(annotation_path):
      raise ValueError('Annotation file %s does not exist' % annotation_path)
    with open(annotation_path, 'r') as json_file:
      annotations = json.load(json_file)
      instance_annotations = annotations['annotations']
      if not instance_annotations:
        raise ValueError('Instance annotations is empty.')
      self.coco_instance_annotations = instance_annotations
      categories = annotations['categories']
      if len(categories) != self.num_all_classes:
        raise ValueError(
            'Total number of MSCOCO classes %d should be equal to the sum of '
            'train, val, test classes %d.' %
            (len(categories), self.num_all_classes))
      self.coco_categories = categories
    self.coco_name_to_category = {cat['name']: cat for cat in categories}

    if box_scale_ratio < 1.0:
      raise ValueError('Box scale ratio must be greater or equal to 1.0.')
    self.classes_per_split = {
        0: 0,
        1: 0,
        2: 0
    }
    self.images_per_class = collections.defaultdict(int)

    # Maps each class id to the name of its class.
    self.class_names = {}

  def get_splits(self):
    """Create splits for MSCOCO and store them in the default path.
    Returns:
      The splits for this dataset, represented as a dictionary mapping each of
      'train', 'valid', and 'test' to a list of class names.
    """
    with open('mscoco_splits.json') as jsonFile:
        split = json.load(jsonFile)
        jsonFile.close()
    return split

  def create_dataset_specification_and_records(self):
    """Implements DatasetConverter.create_dataset_specification_and_records."""
    splits = self.get_splits()

    # Get the names of the classes assigned to each split.
    train_classes = splits['train']
    valid_classes = splits['valid']
    test_classes = splits['test']
    

    self.classes_per_split[0] = len(train_classes)
    self.classes_per_split[1] = len(valid_classes)
    self.classes_per_split[2] = len(test_classes)
    all_classes = list(
        itertools.chain(splits['train'], splits['valid'], splits['test'])) #adapt to metadatasets standarts
    # Map original COCO "id" to class ids that conform to DatasetConverter's
    # contract.
    coco_id_to_class_id = {}
    for class_id, class_name in enumerate(all_classes):
      self.class_names[class_id] = class_name
      category = self.coco_name_to_category[class_name]
      coco_id_to_class_id[category['id']] = class_id

    def get_image_crop_and_class_id_save(annotation):
      """Gets image crop and its class label and saves it"""
      image_id = annotation['image_id']
      image_path = os.path.join(self.image_dir, '%012d.jpg' % image_id)
      # The bounding box is represented as (x_topleft, y_topleft, width, height)
      bbox = annotation['bbox']
      coco_class_id = annotation['category_id']
      class_id = coco_id_to_class_id[coco_class_id]  
      with open(image_path, 'rb') as f:
        # The image shape is [?, ?, 3] and the type is uint8.
        image = Image.open(f)
        image = image.convert(mode='RGB')
        image_w, image_h = image.size

        def scale_box(bbox, scale_ratio):
          x, y, w, h = bbox
          x = x - 0.5 * w * (scale_ratio - 1.0)
          y = y - 0.5 * h * (scale_ratio - 1.0)
          w = w * scale_ratio
          h = h * scale_ratio
          return [x, y, w, h]

        x, y, w, h = scale_box(bbox, self.box_scale_ratio)
        # Convert half-integer to full-integer representation.
        # The Python Imaging Library uses a Cartesian pixel coordinate system,
        # with (0,0) in the upper left corner. Note that the coordinates refer
        # to the implied pixel corners; the centre of a pixel addressed as
        # (0, 0) actually lies at (0.5, 0.5). Since COCO uses the later
        # convention and we use PIL to crop the image, we need to convert from
        # half-integer to full-integer representation.
        xmin = max(int(round(x - 0.5)), 0)
        ymin = max(int(round(y - 0.5)), 0)
        xmax = min(int(round(x + w - 0.5)) + 1, image_w)
        ymax = min(int(round(y + h - 0.5)) + 1, image_h)
        image_crop = image.crop((xmin, ymin, xmax, ymax))
        crop_width, crop_height = image_crop.size
        if crop_width <= 0 or crop_height <= 0:
          raise ValueError('crops are not valid.')
        if not os.path.isdir(os.path.join(self.data_root,'imgs_g',self.class_names[class_id])):
            os.mkdir(os.path.join(self.data_root,'imgs_g',self.class_names[class_id]))
        image_crop.save(os.path.join(self.data_root,'imgs_g',self.class_names[class_id] , '%012d.jpg' % image_id))
      return image_crop, class_id


    for i, annotation in enumerate(tqdm(self.coco_instance_annotations)):
      try:
        image_crop, class_id = get_image_crop_and_class_id_save(annotation)
      except IOError:
        logging.warning('Image can not be opened and will be skipped.')
        continue
      except ValueError:
        logging.warning('Image can not be cropped and will be skipped.')
        continue

      logging.info('writing image %d/%d', i,
                   len(self.coco_instance_annotations))

      # TODO(manzagop): refactor this, e.g. use write_tfrecord_from_image_files.
      #image_crop_bytes = io.BytesIO()
      #image_crop.save(image_crop_bytes, format='JPEG')
      #image_crop_bytes.seek(0)

      self.images_per_class[class_id] += 1


if __name__ == "__main__":
    conv = MSCOCOConverter(os.path.join(args.dataset_path, 'mscoco'))    
    conv.get_splits()
    conv.create_dataset_specification_and_records()
    
