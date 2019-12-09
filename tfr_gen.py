import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import scipy.io as sio

"""Converts the tensor to bytes."""
def _tensor_to_bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

"""Creates tfrecords for given subject."""
def create_tfrecords_with_writer(W, S):
    train_path = '/home/mosam/sandbox/nrsfm/datasets/hand.train'
    with tf.io.TFRecordWriter(train_path) as writer:
        W = W.reshape(-1, 21, 2)
        S = S.reshape(-1, 21, 3)
        print(W.shape)
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

"""Shift the image to the mean/Normalize the image."""
def mean_centering(matrix):    
    matrix[:, 0] = matrix[:, 0] - np.mean(matrix[:, 0])
    matrix[:, 1] = matrix[:, 1] - np.mean(matrix[:, 1])
    if matrix.shape[1] == 3:
        matrix[:, 2] = matrix[:, 2] - np.mean(matrix[:, 2])
    return matrix

"""Specify keypoints, normalize, and generate 2Fx3 / 2Fx2 formatted W and S."""
keypoints = 21
mat_contents = sio.loadmat('../datasets/train_hands.mat')
W, S = mat_contents['W'], mat_contents['S']

"""Generate 2Fx3 / 2Fx2 formatted W and S."""
final_W, final_S = np.zeros((W.shape[0], 2)), np.zeros((S.shape[0], 3))
temp_W = np.zeros((21, 2))
temp_S = np.zeros((21, 3))

for idx in range(int(S.shape[0]/keypoints)):
    temp_W = W[keypoints*(idx): keypoints*(idx+1), :]
    temp_W = mean_centering(temp_W)
    final_W[keypoints*(idx): keypoints*(idx+1), :] = temp_W
    
    temp_S = S[keypoints*(idx): keypoints*(idx+1), :]
    temp_S = mean_centering(temp_S)
    final_S[keypoints*(idx): keypoints*(idx+1), :] = temp_S

"""TFR generation"""
create_tfrecords_with_writer(final_W, final_S)    
