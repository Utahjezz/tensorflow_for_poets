import os, sys

import tensorflow as tf
from scipy import spatial

from sklearn.neighbors import NearestNeighbors

import numpy as np

import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# change this as you see fit
image_path = sys.argv[1]

bottleneck_dir = sys.argv[2]

nneighbors_sel = sys.argv[3]

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile("retrained_labels.txt")]


def select_alg(nneighbors_sel):
    alg_list = {
        "KDTree": 'kd_tree',
        "BallTree": 'ball_tree',
        "Auto": 'auto'
    }
    return alg_list[nneighbors_sel]


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
        features = sess.run(last_layer, {'DecodeJpeg/contents:0': image_data})

        max_similarity = 0

        train_dict = {}

        for category_dir in os.listdir(bottleneck_dir):
            print(category_dir)
            for file in os.listdir(os.path.join(bottleneck_dir, category_dir)):
                with open(os.path.join(os.path.join(bottleneck_dir, category_dir), file), 'rb') as csv_file:
                    string = csv_file.readline()
                    temp = [float(x) for x in string.split(",")]
                    train_list_features.append(temp)
                    index = train_list_features.index(temp)
                    train_dict[index] = {"index": index,
                                         "file": os.path.join(os.path.join(bottleneck_dir, category_dir), file)}

    print("training NearestNeighbors with ", nneighbors_sel)
    X = np.asarray(train_list_features)
    alg = select_alg(nneighbors_sel)
    nbrs = NearestNeighbors(n_neighbors=5, algorithm=alg).fit(X)
    print("training NearestNeighbors")
    print(results)
        for res in results[1][0]:
                print(train_dict[res])
