#####################################################################################################
# testing VGG face model using a pre-trained model
# written by Zhifei Zhang, Aug., 2016
#####################################################################################################

from vgg_face import vgg_face
from imageio import imread
from skimage.transform import resize
from scipy.misc import imresize
import tensorflow as tf
import numpy as np
import sys

# build the graph
graph = tf.Graph()
with graph.as_default():
    input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
    output, average_image, class_names = vgg_face('vgg-face.mat', input_maps)
    # Finds values and indices of the k largest entries
    k = 3
    values, indices = tf.nn.top_k(output['prob'], k)

image_path = 'Aamir_Khan_March_2015.jpg'
if sys.argv != None and len(sys.argv) > 1:
    image_path = sys.argv[1]

print('Detecting Face in Image %s '% (image_path))

# read sample image
img = imread(image_path, pilmode='RGB') #changed to remove deprecation, old call to was to scipy.misc.imread
img = img[0:250, :, :]
img = resize(img, (224, 224)) #replaced to remove deprecation, old call was to imresize
# next 2 are from https://stackoverflow.com/a/44265224
img = 255 * img
img = img.astype(np.uint8)
# originally here
img = img - average_image

# run the graph
with tf.Session(graph=graph) as sess:
    # testing on the sample image
    [prob, ind, out] = sess.run([values, indices, output], feed_dict={input_maps: [img]})
    print(prob, ind)
    prob = prob[0]
    ind = ind[0]
    print('\nClassification Result:')
    for i in range(k):
        print('\tCategory Name: %s \n\tProbability: %.2f%%\n' % (class_names[ind[i]][0][0], prob[i]*100))
    sess.close()

