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

import tensorflow_datasets.public_api as tfds
import tensorflow as tf


# TODO(rtt_news_dataset): BibTeX citation
_CITATION = """
"""

# TODO(rtt_news_dataset):
_DESCRIPTION = """
"""


class RttNewsDataset(tfds.core.GeneratorBasedBuilder):
  """Financial news from https://www.rttnews.com/. Dated 2015-02-19 to 2020-03-02 ."""

  # TODO(rtt_news_dataset): Set up version.
  VERSION = tfds.core.Version('0.1.0')

  MANUAL_DOWNLOAD_INSTRUCTIONS = "lmao"

  def _info(self):
    # TODO(rtt_news_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            "article": tfds.features.Text(),
            "label": tfds.features.Tensor(shape=(), dtype=tf.float32),
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=("article", "label"),
        # Homepage of the dataset for documentation
        homepage='https://www.rttnews.com/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # TODO(rtt_news_dataset): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    data_path = dl_manager.manual_dir
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                "article_text_path": data_path
            },
        ),
    ]

  def _generate_examples(self, article_text_path):
    """Yields examples."""
    # TODO(rtt_news_dataset): Yields (key, example) tuples from the dataset
    for article_file in tf.io.gfile.listdir(article_text_path):
        file_path = os.path.join(article_text_path, article_file)
        file_ptr = tf.io.gfile.GFile(file_path)
        label = file_ptr.readline()
        yield article_file, {
            "article": file_ptr.read(),
            "label": label
        }


