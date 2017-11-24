#/usr/bin/env python
#coding:UTF-8

import numpy as np
import tensorflow as tf
#假设bottleneck是n*n*d维的张量，则weight_mat就是(1,n^2)维的向量
def cal_attented_tensor(bottleneck, weight_mat):
	print('#########')
	shape_bott=bottleneck.shape
	batch_size=shape_bott[0]
	height=shape_bott[1]
	width=shape_bott[2]
	num_channels=shape_bott[3]

	#shape_wt=weight_mat.shape
	#weight_mat_lenth=shape_wt[0]
	
	bottleneck=tf.reshape(bottleneck,[int(batch_size), int(height*width), int(num_channels)])
	#weight_mat=tf.reshape(weight_mat,[1,-1])
	#为方便计算，用weight_mat乘bottleneck
	bottleneck=tf.transpose(bottleneck,[0,2,1])
	weight_mat=tf.transpose(weight_mat,[1,0])
	print(bottleneck.shape)
	#加权，一步到位
	result=tf.convert_to_tensor( [tf.matmul(bottleneck[i],weight_mat) for i in range(batch_size)]) #对于batch中的每一条数据来说，应该是个列向量，因此还需要再做一次转置
	return tf.reshape(result,[int(batch_size), int(num_channels)])

if __name__=='__main__':
	bottleneck1=tf.constant([1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6],shape=[2,2,2,2])
	weight_mat1=tf.constant([1,2,3,4],shape=[1,4])
	
	result= cal_attented_tensor(bottleneck1,weight_mat1)
	with tf.Session() as sess:
		#print(sess.run(bottleneck))
		#print(sess.run(weight_mat))
	
		print(sess.run(result))	
		print(sess.run(result).shape)	
