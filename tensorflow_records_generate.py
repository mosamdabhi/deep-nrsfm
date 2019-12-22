import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import scipy.io as sio

"""Converts the tensor to bytes."""
def _tensor_to_bytes_feature(value):
   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

"""Creates tfrecords for given subject."""
def create_tfrecords_with_writer():
   train_path = '/home/mosam/sandbox/nrsfm/datasets/hand.train'
   with tf.io.TFRecordWriter(train_path) as writer:
       S = sio.loadmat('S_hand_train.mat')       
       S = S['final_S']

       W = sio.loadmat('W_hand_train.mat')       
       W = W['final_W']       

       S = S.reshape(-1, 21, 3)
       W = W.reshape(-1, 21, 2)              

       for i in range(S.shape[0]):            
          pts3d = S[i,:].reshape(21,3)                       
          pts2d = W[i,:].reshape(21,2)                    
          points3d_raw = pts3d.astype("float32").tostring()
          points2d_raw = pts2d.astype("float32").tostring()
          example = tf.train.Example(
            features=tf.train.Features(
              feature={
              "points3d_raw": _tensor_to_bytes_feature(points3d_raw),
              "points2d_raw": _tensor_to_bytes_feature(points2d_raw),
              }
              )
            )
          writer.write(example.SerializeToString())

create_tfrecords_with_writer()