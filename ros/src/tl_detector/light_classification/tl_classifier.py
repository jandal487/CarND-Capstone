from styx_msgs.msg import TrafficLight
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        pCount = 50
        
        thdMin = np.array([0, 100, 100], np.uint8)
        thdMax = np.array([10, 255, 255], np.uint8)        
        binImg1 = cv2.inRange(hsvImg, thdMin, thdMax)

        thdMin = np.array([160, 100, 100], np.uint8)
        thdMax = np.array([180, 255, 255], np.uint8)
        binImg2 = cv2.inRange(hsvImg, thdMin, thdMax)
        
        pSum = cv2.countNonZero(binImg1) + cv2.countNonZero(bingIm2)
        if  pSum > pCount:
            return TrafficLight.RED
        
        return TrafficLight.UNKNOWN
