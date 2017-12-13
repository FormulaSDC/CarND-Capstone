from styx_msgs.msg import TrafficLight
import numpy as np
import rospy
import tensorflow as tf
import cv2
import os



class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        #rospy.loginfo("current dir is  %s ", os.path.realpath('.'))

        tf.reset_default_graph()
        config = tf.ConfigProto(
           gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4),
           device_count = {'GPU': 1}
        )
        
        self.session = tf.Session(config=config);

        save_file = './light_classification/mixNetI-1.ckpt.meta'
        saver = tf.train.import_meta_graph(save_file)
        saver.restore(self.session,tf.train.latest_checkpoint('./light_classification/'))

        graph = tf.get_default_graph()
        self.batch_x =  graph.get_tensor_by_name("batch_x:0")
        self.keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self.out = graph.get_tensor_by_name("out:0")

        #fc2 = MixNet(batch_x)

        #saver = tf.train.Saver();
        #saver.restore(self.session, save_file)
        self.ans = tf.argmax(self.out, 1);

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        #convert from BGR to RGB and normalize
        img = (image.astype('float32')-127.)/255.;
        a = self.session.run([self.ans], feed_dict={self.batch_x: [img],
                                                    self.keep_prob: 1.0})

        prediction = a[0]
          
        return prediction
