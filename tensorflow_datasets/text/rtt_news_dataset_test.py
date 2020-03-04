"""TODO(rtt_news_dataset): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys

root_path = os.path.abspath(__file__)
while os.path.basename(root_path) != "tensorflow_datasets":
    print(root_path)
    root_path = os.path.dirname(root_path)
root_path = os.path.dirname(root_path)
print(root_path)
sys.path.append(root_path)

from tensorflow_datasets import testing
from tensorflow_datasets.text import rtt_news_dataset
import tensorflow as tf


class RttNewsDatasetTest(testing.DatasetBuilderTestCase):
  # TODO(rtt_news_dataset):
  DATASET_CLASS = rtt_news_dataset.RttNewsDataset
  SPLITS = {
      "train": 4,  # Number of fake train example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}
  DL_EXTRACT_RESULT = {
    r'2015-02-19_Caution%20Prevails%20As%20Greek%20Crisis%20Continues.txt': r'2015-02-19_Caution%20Prevails%20As%20Greek%20Crisis%20Continues.txt',
    r'2015-02-19_Stocks%20May%20Continue%20To%20Experience%20Choppy%20Trading%20-%20U.S.%20Commentary.txt': r'2015-02-19_Stocks%20May%20Continue%20To%20Experience%20Choppy%20Trading%20-%20U.S.%20Commentary.txt',
    r'2015-02-19_Wall%20Street%20Seen%20Lower%20Amid%20Continuing%20Uncertainty.txt': r'2015-02-19_Wall%20Street%20Seen%20Lower%20Amid%20Continuing%20Uncertainty.txt',
    r'2015-02-20_French%20Market%20Slides%20Amid%20Greek%20Concerns.txt': r'2015-02-20_French%20Market%20Slides%20Amid%20Greek%20Concerns.txt',
  }


if __name__ == "__main__":
  testing.test_main()

