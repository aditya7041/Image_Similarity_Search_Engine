"""
 Deep learning network(CNN) to get similar images
 Hack Week Project !!
 Written by : adityag@zillowgroup.com
"""
import tensorflow as tf
from boto.s3.connection import S3Connection
from PIL import Image
import numpy as np
import glob
import math 


N = 224
n_input = N * N * 3
n_classes = 2
step = 1
dropout = 0.85  # Dropout, probability to keep unit
num_test_images = 1

# tf Graph input
#x = tf.placeholder(tf.float32, [None, n_input])
#y = tf.placeholder(tf.float32, [None, n_classes])

keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

def get_an_image_array(image_path):
    orig_image = Image.open(image_path).resize((224,224),Image.ANTIALIAS)
    np_image = np.array(orig_image)
    reshape_image = np_image.reshape((1,224*224*3))
    image = reshape_image.astype(np.float32)*1.0/255.0
    return image

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Get the feature_vector of an image on a trained network 
def get_image_feature_vector(rank_test_x, weights, biases,fc_feature_vec):

    print('Getting image feature vector on trained network : ')
    rank_test_x = tf.reshape(rank_test_x, shape=[-1, N, N, 3])

    conv1 = conv2d(rank_test_x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv4 = maxpool2d(conv4, k=2)

    fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1']) 
    fc_feature_vec = fc1

    return fc1

def cosine_similarty(v1,v2):
     # "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0.0, 0.0, 0.0

    for i in range(len(v1[0])):
        x = v1[0][i]; 
        y = v2[0][i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y

    return sumxy/math.sqrt(sumxx*sumyy)

def print_step(step):
    print("step = ", step)

weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'wd1': tf.Variable(tf.random_normal([14 * 14 * 128, 4096])),
    'out': tf.Variable(tf.random_normal([4096, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([128])),
    'bc4': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Ranking related variables
rank_test_x = tf.placeholder(tf.float32, [num_test_images, n_input])
fc_feature_vec = tf.Variable(tf.random_normal([1,4096]))
get_fc_features = get_image_feature_vector(rank_test_x,weights,biases,fc_feature_vec)
    
# Initializing the variables
init = tf.initialize_all_variables()
saver = tf.train.Saver(tf.all_variables())


def similarity_search():
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        print(' ')
        print('Running Image Similarity Engine :')
        print('Loading Tensorflow deep learning model...\n')
        saver.restore(sess,'/Users/adityag/Desktop/HackWeek/tf_models/thursday_afternoon_model')

        threshold = .749
        below_threshold = False
        num_best_images_to_return = 6
        best_images_list = []
        thresh_str = '<--------------BELOW THRESHOLD IMAGES------------->'
        
        #print(' main file path :',FLAGS.main_image)
        #print(' candidate image path : ',FLAGS.candidate_path)
        cand_path =FLAGS.candidate_path + '/*.jpg'
        # look for each image in the main folder:
        for main_image_path in glob.iglob(FLAGS.main_image):
            main_image_array = get_an_image_array(main_image_path)
            main_vc = sess.run(get_fc_features,feed_dict={rank_test_x: main_image_array})
            main_image_name = main_image_path.split('/')[len(main_image_path.split('/'))-1]
            print('Finding images similar to : ', main_image_name)

            tuple_list = []
            for image_path in glob.iglob(cand_path):
                orig_image = Image.open(image_path).resize((224,224), Image.ANTIALIAS)
                np_image = np.array(orig_image)
                reshape_image = np_image.reshape((1,224*224*3))
                image_array = reshape_image.astype(np.float32)*1.0/255.0
                cand_vc = sess.run(get_fc_features,feed_dict={rank_test_x: image_array}) 
                corr_score = cosine_similarty(cand_vc,main_vc)
                str_image = str(image_path)
                image_name = str_image.split('/')[len(str_image.split('/'))-1]
                tuple_list.append((image_name,corr_score))
            
            # Sorting the results, writing to a file and printing it
            tuple_list.sort(key=lambda tup: tup[1], reverse=True)
            list_len = len(tuple_list)
            f= open("/Users/adityag/Desktop/HackWeek/Demo/image_similarity_engine/ranking_output.txt","w")
            
            for i in range(0,num_best_images_to_return):
                f.write(str(tuple_list[i][0]) + ':'  + str(tuple_list[i][1])  + '\n')
            f.close()
            
            f1= open("/Users/adityag/Desktop/HackWeek/Demo/image_similarity_engine/reverse_ranking_output.txt","w")
            for i in range(0,5):
                f1.write(str(tuple_list[list_len-1-i][0]) + ':'  + str(tuple_list[list_len-1-i][1])  + '\n')
            f1.close()
            
            print('\nImage name           Score \n')
            for tup in tuple_list:
                if tup[1]==1.0:
                    printf('Exact match found')
                if (below_threshold==False) & (tup[1]<threshold):
                    print(thresh_str)
                    below_threshold = True
                if len(str(tup[0])) == 14:
                    print_str = str(tup[0]) + '       score : ' + str(tup[1])
                else:
                    print_str = str(tup[0]) + '        score : ' + str(tup[1])
                print(print_str)
            print('<-----------DONE WITH RANKING OF ALL IMAGES GIVEN BY WEB CRAWLER------->')


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('main_image', '',
                           """Path of main image """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('candidate_path', '',
                           """Path of candidate images """
                           """and checkpoint.""")


def main(argv=None):  # pylint: disable=unused-argument
    similarity_search()

if __name__ == '__main__':
  tf.app.run()
