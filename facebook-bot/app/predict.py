import cv2
import numpy as np
import tf_serving_client

input_size = 608
max_box_per_image = 4
# TODO: Set correct anchors 
anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

labels   = ['Newsfeed item', 'In-feed AD', 'Side AD']

nb_class = len(labels)
nb_box   = 3
class_wt = np.ones(nb_class, dtype='float32')


class BoundBox:
    def __init__(self, x, y, w, h, c = None, classes = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    
    if np.min(x) < t:
        x = x/np.min(x)*t
        
    e_x = np.exp(x)
    
    return e_x / e_x.sum(axis, keepdims=True)

def bbox_iou(box1, box2):
    x1_min  = box1.x - box1.w/2
    x1_max  = box1.x + box1.w/2
    y1_min  = box1.y - box1.h/2
    y1_max  = box1.y + box1.h/2
    
    x2_min  = box2.x - box2.w/2
    x2_max  = box2.x + box2.w/2
    y2_min  = box2.y - box2.h/2
    y2_max  = box2.y + box2.h/2
    
    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])
    
    intersect = intersect_w * intersect_h
    
    union = box1.w * box1.h + box2.w * box2.h - intersect
    
    return float(intersect) / union
    
def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3 

def normalize(image):
    return image / 255.

def decode_netout(netout, obj_threshold=0.3, nms_threshold=0.01):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []
    
    # decode the output by the network
    netout[..., 4]  = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold
    
    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]
                
                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]
                    
                    box = BoundBox(x, y, w, h, confidence, classes)
                    
                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
                        
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    
    return boxes

def draw_boxes(image, boxes, labels):
    
    image_h, image_w, _ = image.shape
    
    for box in boxes:
        xmin  = int((box.x - box.w/2) * image.shape[1])
        xmax  = int((box.x + box.w/2) * image.shape[1])
        ymin  = int((box.y - box.h/2) * image.shape[0])
        ymax  = int((box.y + box.h/2) * image.shape[0])

        boxColor = (0,0,255)
        foreColor = (0,255,255)

        # First label is for newsfeed
        if (box.get_label() == 0):
            boxColor = (0,255,0)
            foreColor = (255,255,255)
        
        text = labels[box.get_label()] + ' ' + str(box.get_score())
        textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                    1e-3 * image_h, 2)[0]

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), boxColor, 4)
        cv2.rectangle(image, (xmax - textSize[0] - 20,ymin), (xmax,ymin - 35), boxColor, -1)
        cv2.putText(image, 
                    text, 
                    (xmax - textSize[0] - 10, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image_h,
                    foreColor, 2)
        
    return image        
        

def predict_and_draw_boxes(image_data):
    image_decoded = cv2.imdecode(np.asarray(bytearray(image_data), dtype=np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image_decoded, (input_size, input_size))
    image = normalize(image)

    input_image = image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)
    dummy_array = dummy_array = np.zeros((1,1,1,1,max_box_per_image,4))

    print(input_image.shape)
    print(dummy_array.shape)
    
    netout = tf_serving_client.make_prediction([input_image, dummy_array])

    print(netout.shape)

    boxes  = decode_netout(netout)
    orig_image = cv2.imdecode(np.asarray(bytearray(image_data), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    image = draw_boxes(orig_image, boxes, ["Newsfeed", "AD", "Side Ad"])

    ret, pngBytes = cv2.imencode(".jpg", image)
    return (pngBytes, boxes)

