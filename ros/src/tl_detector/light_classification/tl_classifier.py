from styx_msgs.msg import TrafficLight
import numpy as np
import rospy

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.upper_thr = .3
        self.lower_thr = .05
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        # Cropping the image edges a bit since the bounding box is slighly larger than the actual traffic light
        image = image[int(0.05 * image.shape[0]) : int(0.95 * image.shape[0]), 
                      int(0.05 * image.shape[1]) : int(0.95 * image.shape[1]), :]

        blue_img    = image[:,:,0]
        red_img     = image[:,:,1]
        green_img   = image[:,:,2]
        red_layer   = np.sum(red_img   > .8*red_img.max())   / float(red_img.size)
        green_layer = np.sum(green_img > .8*green_img.max()) / float(red_img.size)
        blue_layer  = np.sum(blue_img  > .8*blue_img.max())  / float(red_img.size)

        rospy.loginfo("Red: {}, Green: {}, Blue: {}, Shape: {}".format(red_layer, green_layer, blue_layer, red_img.size))


        prediction = TrafficLight.UNKNOWN

        if red_layer >= self.upper_thr and green_layer <= self.lower_thr:
          prediction = TrafficLight.RED
        elif red_layer <= self.lower_thr and green_layer <= self.lower_thr:
          prediction = TrafficLight.YELLOW
        elif green_layer >= self.upper_thr:
          prediction = TrafficLight.GREEN
        else:
          prediction = TrafficLight.UNKNOWN
          
        return prediction
