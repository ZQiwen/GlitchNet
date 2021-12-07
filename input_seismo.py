# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 17:35:10 2021

@author: Chiween


DESCRIPTION:
    Functions for reading seismo data.
"""

import numpy as np
import tensorflow as tf
import os

class DataSet(object):
  
  def __init__(self, seismo, labels, sp_data=False, dtype=tf.float32):
    """Construct a DataSet.
    
      `one_hot` arg is used only if fake_data is true.  
      `dtype` can be either `int32` to leave the input as raw counts, 
        or `float32` to rescale into `[-1., 1.]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.int32, tf.float32):
      raise TypeError('Invalid seimio dtype %r, expected int32 or float32' %
                      dtype)

    
    assert seismo.shape[0] == labels.shape[0], (
        'seismo.shape: %s labels.shape: %s' % (seismo.shape, labels.shape))
    self._num_examples = seismo.shape[0]
    
    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    # assert seismo.shape[3] == 1

    if dtype == tf.float32:
      # Convert from raw counts -> [-1.0, 1.0].
      seismo = seismo.astype(np.float32)
      seismo = np.apply_along_axis(lambda x: x-np.linspace(np.mean(x[:20]),np.mean(x[-20:]),x.shape[0]), 1, seismo)
      seismo = np.apply_along_axis(lambda x: x/abs(x).max(), 1, seismo)
      
    self._seismo = seismo
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    
    
  @property
  def seismos(self):
    return self._seismo

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed
  
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed = 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._seismo = self._seismo[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._seismo[start:end], self._labels[start:end]
    
def read_data_sets(train_dir, sp_data=False, dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets=DataSets()

  if sp_data:
    TRAIN_SEISMOS = 'pureglitch-sp.npy'
    TRAIN_LABELS = 'pureglitch-sp.npy'
    TEST_SEISMOS = 'pureglitch-sp.npy'
    TEST_LABELS = 'pureglitch-sp.npy'
    
  else:
      
    TRAIN_SEISMOS = 'hq_pure_glitch.npy'
    TRAIN_LABELS = 'hq_pure_glitch.npy'
    TEST_SEISMOS = 'hq_pure_glitch.npy'
    TEST_LABELS = 'hq_pure_glitch.npy'
  
  VALIDATION_SIZE = 1000
  
  train_seismos = np.load(train_dir+os.path.sep+TRAIN_SEISMOS)
  
  if not sp_data:
    blacklist = np.loadtxt(
      r'data\hq_pure_glitch_blacklist.txt', dtype=int)
    train_seismos = np.delete(train_seismos, blacklist, axis=1)
    train_seismos = train_seismos.T
  
  train_labels = train_seismos
  
  validation_seismos = train_seismos[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_seismos = train_seismos[VALIDATION_SIZE:-VALIDATION_SIZE]
  train_labels = train_labels[VALIDATION_SIZE:-VALIDATION_SIZE]
  test_seismos = train_seismos[-VALIDATION_SIZE:]
  test_labels = train_labels[-VALIDATION_SIZE:]

  data_sets.train = DataSet(train_seismos, train_labels, dtype=dtype)
  data_sets.validation = DataSet(validation_seismos, validation_labels,
                                 dtype=dtype)
  data_sets.test = DataSet(test_seismos, test_labels, dtype=dtype)

  return data_sets

















































