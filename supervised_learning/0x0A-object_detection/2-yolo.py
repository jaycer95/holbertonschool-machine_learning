#!/usr/bin/env python3
""" Object Detection """

from tensorflow import keras as K
import numpy as np


class Yolo:
    """ Use the Yolo v3 algorithm to perform object detection """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ Use the Yolo v3 algorithm to perform object detection """
        self.model = K.models.load_model(filepath=model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, z):
        """ sigmoid function"""
        return (1 / (1 + np.exp(-z)))

    def process_outputs(self, outputs, image_size):
        """ process outputs """
        boxes = []
        box_confidences = []
        box_class_probs = []
        img_h, img_w = image_size

        for output in outputs:
            boxes.append(output[..., 0:4])
            box_confidences.append(self.sigmoid(output[..., 4, np.newaxis]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))
        for i, box in enumerate(boxes):
            gr_h, gr_w, anchors_boxes, _ = box.shape
            cx = np.indices((gr_h, gr_w, anchors_boxes))[1]
            cy = np.indices((gr_h, gr_w, anchors_boxes))[0]
            t_x = box[..., 0]
            t_y = box[..., 1]
            t_w = box[..., 2]
            t_h = box[..., 3]
            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]
            bx = (self.sigmoid(t_x) + cx) / gr_w
            by = (self.sigmoid(t_y) + cy) / gr_h
            bw = (np.exp(t_w) * p_w) / self.model.input.shape[1].value
            bh = (np.exp(t_h) * p_h) / self.model.input.shape[2].value
            tl_x = bx - bw / 2
            tl_y = by - bh / 2
            br_x = bx + bw / 2
            br_y = by + bh / 2
            box[..., 0] = tl_x * img_w
            box[..., 1] = tl_y * img_h
            box[..., 2] = br_x * img_w
            box[..., 3] = br_y * img_h
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """filter boxes"""
        box_scores = [box_confidences[i] * box_class_probs[i]
                      for i in range(len(box_confidences))]
        box_classes = [np.argmax(box_class, axis=-1)
                       for box_class in box_scores]
        box_class_scores = [np.max(box_scr, axis=-1) for box_scr in box_scores]
        prediction_mask = [np.where(bcs >= self.class_t)
                           for bcs in box_class_scores]

        boxes = [box[mask] for box, mask in zip(boxes, prediction_mask)]
        scores = [box[mask] for box, mask in zip(
            box_class_scores,
            prediction_mask)]
        classes = [box[mask]
                   for box, mask in zip(box_classes, prediction_mask)]
        classes = np.concatenate(classes).reshape(-1)
        boxess = np.concatenate(boxes)
        scores = np.concatenate(scores).reshape(-1)
        return boxess, classes, scores
