import os

import cv2
import numpy as np
import rospy
import tensorflow as tf
from styx_msgs.msg import TrafficLight


class TLClassifier(object):
    def __init__(self, model_file):
        # TODO load classifier
        self.current_light = TrafficLight.UNKNOWN
        cwd = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cwd, "train_model/{}".format(model_file))

        # load frozen tensorflow model
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        #getting all the available names in tensor to avoid an error.<Reference> https://stackoverflow.com/questions/35336648/list-of-tensor-names-in-graph-in-tensorflow/35337827
        ops = self.detection_graph.get_operations()
        self.all_tensor_names = {output.name for op in ops for output in op.outputs}

        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        if 'image_tensor:0' in self.all_tensor_names:
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        if 'detection_boxes:0' in self.all_tensor_names:
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        if 'detection_scores:0' in self.all_tensor_names:
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        if 'detection_classes:0' in self.all_tensor_names:
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        if 'num_detections:0' in self.all_tensor_names:
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # return TrafficLight.RED
        # TODO implement light color prediction
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (im_width, im_height, _) = image_rgb.shape
        image_np = np.expand_dims(image_rgb, axis=0)

        # Actual detection.
        if None not in [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections, self.sess]:
            with self.detection_graph.as_default():
                (boxes, scores, classes, num) = self.sess.run(
                    [self.detection_boxes, self.detection_scores,
                    self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: image_np})

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)
            
            min_score_thresh = .5
            detected = classes[0] if (num > 0 and scores[0] > min_score_thresh) else -1
            if detected == 1:
                print('GREEN')
                return TrafficLight.GREEN
            if detected == 2:
                print('RED')
                return TrafficLight.RED
            if detected == 3:
                print('YELLOW')
                return TrafficLight.YELLOW
            else:
                return TrafficLight.UNKNOWN

        return TrafficLight.UNKNOWN       
