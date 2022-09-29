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



def write_from_npy_single_channel(class_npy_file, class_label,
                                           new_path):
  """Create and write a tf.record file for the data of a class.
  This assumes that the provided .npy file stores the data of a given class in
  an array of shape [num_images_of_given_class, side**2].
  In the case of the Quickdraw dataset for example, side = 28.
  Each row of that array is interpreted as a single-channel side x side image,
  read into a PIL.Image, converted to RGB and then written into a record.
  Args:
    class_npy_file: the .npy file of the images of class class_label.
    class_label: the label of the class that a Record is being made for.
    output_path: the location to write the Record.
  Returns:
    The number of images in the .npy file for class class_label.
  """

  def load_image(img):
    """Load image img.
    Args:
      img: a 1D numpy array of shape [side**2]
    Returns:
      a PIL Image
    """
    # We make the assumption that the images are square.
    side = int(np.sqrt(img.shape[0]))
    # To load an array as a PIL.Image we must first reshape it to 2D.
    img = Image.fromarray(img.reshape((side, side)))
    img = img.convert('RGB')
    return img

  imgs = np.load(class_npy_file)

  # If the values are in the range 0-1, bring them to the range 0-255.
  if imgs.dtype == np.bool:
    imgs = imgs.astype(np.uint8)
    imgs *= 255

  # Takes a row each time, i.e. a different image (of the same class_label).
  for i,image in enumerate(imgs):
    img = load_image(image)
    # Compress to JPEG before writing
    #buf = io.BytesIO()
    #img.save(buf, format='JPEG')
    #buf.seek(0)
    write_example(img, class_label,i,new_path)

  return len(imgs)



class QuickdrawConverter():
  """Prepares Quickdraw as required to integrate it in the benchmark."""
  def __init__(self):

    self.data_root = os.path.join(args.dataset_path, 'quickdraw')
    self.classes_per_split = {
        0: 0,
        1: 0,
        2: 0
    }
    self.class_names = {}
    self.images_per_class = collections.defaultdict(int)
  


  def get_splits(self,json_path = 'quickdraw_splits.json'):
    with open(json_path) as jsonFile:
        split = json.load(jsonFile)
        jsonFile.close()
    return split



  def parse_split_data(self, split, split_class_names):
    """Parse the data of the given split.
    Specifically, update self.class_names, self.images_per_class, and
    self.classes_per_split with the information for the given split, and
    create and write records of the classes of the given split.
    Args:
      split: an instance of learning_spec.Split
      split_class_names: the list of names of classes belonging to split
    """
    for class_name in split_class_names:
      self.classes_per_split[split] += 1
      class_label = len(self.class_names)


      # The names of the files in self.data_root for Quickdraw are of the form
      # class_name.npy, for example airplane.npy.
      class_npy_fname = class_name + '.npy'
      self.class_names[class_label] = class_name
      class_path = os.path.join(self.data_root, class_npy_fname)
      # Create and write the tf.Record of the examples of this class.
      num_imgs = write_from_npy_single_channel(class_path, class_label,
                                                        new_path = os.path.join(self.data_root,'all_samples3' ))
      self.images_per_class[class_label] = num_imgs

  def create_dataset_specification_and_records(self):
    """Implements DatasetConverter.create_dataset_specification_and_records.
    If no split file is provided, and the default location for Quickdraw splits
    does not contain a split file, splits are randomly created in this
    function using 70%, 15%, and 15% of the data for training, validation and
    testing, respectively, and then stored in that default location.
    The splits for this dataset are represented as a dictionary mapping each of
    'train', 'valid', and 'test' to a list of class names. For example the value
    associated with the key 'train' may be ['angel', 'clock', ...].
    """

    splits = self.get_splits()
    # Get the names of the classes assigned to each split.
    train_classes = splits['train']
    valid_classes = splits['valid']
    test_classes = splits['test']

    self.classes_per_split[0] = len(train_classes)
    self.classes_per_split[1] = len(valid_classes)
    self.classes_per_split[2] = len(test_classes)

    self.parse_split_data(0, train_classes)
    self.parse_split_data(1, valid_classes)
    self.parse_split_data(2, test_classes)


if __name__ == "__main__":
    conv = QuickdrawConverter()
    conv.get_splits()
    conv.create_dataset_specification_and_records()
