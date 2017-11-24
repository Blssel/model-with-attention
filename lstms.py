#/usr/bin/python
#coding:UTF-8

import tensorflow as tf
import inception_v1
import next_batch


LSTM_HIDDEN_SIZE=200
BATCH_SIZE=10
NUM_STEPS=10000
NUM_LAYERS=3
NUM_CLASSES=101
FRAME_SIZE=224
INCEPTION_OUT_SIZE=7
INCEPTION_OUT_DEPTH=1024
TFRECORD_PATH='/home/mcger/datasets/ucf101/'

def main():
	'''
	#定义输入
	input=tf.placeholder()
	groundtruth=tf.placeholder()
	'''
	#改变打法，输入无需占位符，而且只要提供一个输入即可（类似mapreduce，只要提供一个读取batch的方法剩下的事情框架来处理i）
	video_batch,label=next_batch.next_batch(BATCH_SIZE, RFRECORD_PATH, FRAMES_NEEDED, FRAME_GAP, INPUT_FRAME_SIZE, NUM_CLASSES, category='train')

	#定义3层LSTM，并初始化为0状态
	one_lstm=rnn.cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)
	multilayer_lstm=tf.nn.rnn_cell.MultiRNNCell([one_lstm]*NUM_LAYERS)
	#初始化c和h最初的状态，也就是全零的向量
	initial_state=multilayer_lstm.zero_state(BATCH_SIZE,tf.float32)
	state=initial_state

	with tf.variable_scope("RNN"):
		#定义并初始化attention层参数
		w_att=tf.Variable(tf.truncated_normal([LSTM_HIDDEN_SIZE, INCEPTION_OUT_SIZE*INCEPTION_OUT_SIZE], stddev=0.001))
		#初始化权重矩阵(实际上是个向量)
		weight_mat=tf.constant([1]*49,dtype=tf.float32)
		weight_mat=tf.nn.softmax(weight_mat)
		#定义输出层全连接层参数
		w_fc_out=tf.Variable(tf.truncated_normal([LSTM_HIDDEN_SIZE,NUM_CLASSES],stddev=0.001))
		biases_fc_out=tf.Variable(tf.zeros([NUM_CLASSES]))
		#attention层全连接参数
		w_fc_att=tf.Variable(tf.truncated_normal([LSTM_HIDDEN_SIZE,INCEPTION_OUT_SIZE*INCEPTION_OUT_SIZE],stddev=0.001))
		biases_fc_att=tf.Variable(tf.zeros([INCEPTION_OUT_SIZE*INCEPTION_OUT_SIZE]))
		
		loss=0
		global_step= tf.Variable(0,name='global_step',trainable=False)
		#循环输入
		for time_step in range(NUM_STEPS):
			#想办法让这些神经元共享变量
			if time_step>0: 
				tf.get_variable_scope().reuse_variables()

			inception_output, _ =inception_v1.inception_v1_base(video_batch,final_endpoint='Mixed_5c',scope='InceptionV1')
			#计算wa
			wa=cal_attented_tensor(inception_output, weight_mat)

			#inception_output=tf.reshape([BATCH_SIZE, INCEPTION_OUT_SIZE*INCEPTION_OUT_SIZE, INCEPTION_OUT_SIZE*INCEPTION_OUT_DEPTH], inception_output)
			#wa=tf.mul(weight_mat,inception_out)    #!!!!!!!!!!!!!????????????????????????????
			lstm_output,state=multilayer_lstm(wa,state)
			#计算下一次迭代的权重矩阵
			weight_mat=tf.matmul(state,w_fc_att)       #如何获得h？？？？？？？？？
			#输出
			logits=tf.matmul(lstm_output,w_fc)+b_fc
			#计算损失
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=groundtruth)
			loss+=cross_entropy

		#优化
		train_step = tf.train.MomentumOptimizer(0.5,0.9).minimize(loss, global_step=global_step,var_list=tf.global_variables())

	with tf.Session() as sess:
		tf.global_variable_initializer().run()
		coord=tf.train.Coordinator()
		threads=tf.train.start_queue_runners(sess=sess,coord=coord)
		#循环地训练神经网络
		for i in range(TRAINING_STEPS):
			if i%1000==0:
				loss=sess.run(loss)
				print("After %d training step(s),loss is %g "%(i,loss))
			sess.run(train_step)
		coord.request_stop()
		coord.join(threads)
