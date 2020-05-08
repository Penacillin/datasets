"""TODO(rtt_news_dataset): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import pandas as pd
import datetime
import math

root_path = os.path.abspath(__file__)
while os.path.basename(root_path) != "tensorflow_datasets":
    root_path = os.path.dirname(root_path)
root_path = os.path.dirname(root_path)
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
  VERSION = tfds.core.Version('0.2.0')

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
            "date": tfds.features.Tensor(shape=(), dtype=tf.int64),
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
    price_df = pd.read_csv(os.path.join(data_path, 'market_data', 'GSPC.csv'))
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    price_df.set_index('Date', inplace=True)
    date_split = datetime.date(2019, 2, 1)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "article_text_path": data_path,
                "price_df": price_df.loc[:date_split],
                "date_split": date_split,
                "split_before": True
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                "article_text_path": data_path,
                "price_df": price_df.loc[date_split + datetime.timedelta(1):],
                "date_split": date_split,
                "split_before": False
            },
        )
    ]

  def _generate_examples(self, article_text_path, price_df, date_split, split_before):
    """Yields examples."""
    # TODO(rtt_news_dataset): Yields (key, example) tuples from the dataset
    for article_file in tf.io.gfile.listdir(article_text_path):
        file_path = os.path.join(article_text_path, article_file)
        if tf.io.gfile.isdir(file_path): continue
        file_ptr = tf.io.gfile.GFile(file_path)

        label = 0

        article_date = datetime.date.fromisoformat(article_file[:10])
        desired_date = article_date + datetime.timedelta(1)

        if ((split_before and desired_date > date_split) or
            (not split_before and desired_date < split_before)):
            continue

        desired_date_pdt = pd.Timestamp(desired_date)
        while desired_date_pdt not in price_df.index and desired_date < datetime.date(2100, 1, 1):
            desired_date += datetime.timedelta(days=1)
            desired_date_pdt = pd.Timestamp(desired_date)

        if desired_date_pdt in price_df.index:
            row = price_df.loc[desired_date_pdt]
            label = math.log(row['Close'] / row['Open'])

        yield article_file, {
            "article": file_ptr.read(),
            "date": int(datetime.datetime.combine(
                article_date, datetime.datetime.min.time()).timestamp()),
            "label": label
        }
        file_ptr.close()
