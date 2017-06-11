import os, sys

import tensorflow as tf
from scipy import spatial

from sklearn.neighbors import KDTree

import numpy as np

import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# change this as you see fit
image_path = sys.argv[1]

working_dir = sys.argv[2]

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]

if sys.argv[3] == 'Classification':

	# Unpersists graph from file
	with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
	    graph_def = tf.GraphDef()
	    graph_def.ParseFromString(f.read())
	    tf.import_graph_def(graph_def, name='')

	with tf.Session() as sess:
	    # Feed the image_data as input to the graph and get first prediction
	    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
	    
	    predictions = sess.run(softmax_tensor, \
	             {'DecodeJpeg/contents:0': image_data})
	    
	    # Sort to show labels of first prediction in order of confidence
	    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
	    
	    for node_id in top_k:
	        human_string = label_lines[node_id]
	        score = predictions[0][node_id]
	        print('%s (score = %.5f)' % (human_string, score))
else:
	# Unpersists graph from file
	with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
	    graph_def = tf.GraphDef()
	    graph_def.ParseFromString(f.read())
	    tf.import_graph_def(graph_def, name='')

	with tf.Session() as sess:
	    # Feed the image_data as input to the graph and get first prediction
	    last_layer = sess.graph.get_tensor_by_name('pool_3:0')

	    train_list_features = []
	    features = sess.run(last_layer, {'DecodeJpeg/contents:0' : image_data})

	    max_similarity = 0

	    train_dict = {}

	    for category_dir in os.listdir(working_dir):
	    	print(category_dir)
	    	for file in os.listdir(os.path.join(working_dir, category_dir)):
				with open(os.path.join(os.path.join(working_dir, category_dir), file), 'rb') as csv_file:
					string = csv_file.readline()
					temp = [float(x) for x in string.split(",")]
					train_list_features.append(temp)
					index = train_list_features.index(temp)
					train_dict[index] = {"index": index, "file" : os.path.join(os.path.join(working_dir, category_dir), file)}
						
	print("training KD")
	X = np.asarray(train_list_features)
	kdt = KDTree(X, leaf_size=30)
	print("trained KD")
	query_val = list([float(x) for x in features[0][0][0]])
	print(train_dict[kdt.query(np.asarray(query_val), k=1)[1][0][0]])
		
	  	
	   
	 