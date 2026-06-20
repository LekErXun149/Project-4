import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

CLASSES = ['Foam', 'Lipgloss', 'Acid', 'Control', 'Serum', 'Cream']

class ObjectDetectionDepth(Node):
    def __init__(self):
        super().__init__('object_detection_depth')
        self.bridge = CvBridge()
        self.net = cv2.dnn.readNetFromONNX('/home/group5/best.onnx')
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.latest_depth = None
        
        self.color_sub = self.create_subscription(
            Image, '/camera0/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera0/depth/image_raw', self.depth_callback, 10)
        
        self.result_pub = self.create_publisher(Image, '/detection/image', 10)
        
        self.get_logger().info('Object Detection + Depth node started')

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        outputs = np.array(outputs[0]).T
        boxes, scores, class_ids = [], [], []
        for row in outputs:
            classes_score = row[4:]
            max_score = np.max(classes_score)
            if max_score >= 0.45:
                class_id = np.argmax(classes_score)
                cx, cy, bw, bh = row[:4]
                x = int((cx - bw/2) * w / 640)
                y = int((cy - bh/2) * h / 640)
                width = int(bw * w / 640)
                height = int(bh * h / 640)
                boxes.append([x, y, width, height])
                scores.append(float(max_score))
                class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.45, 0.45)
        results = []
        for i in indices:
            results.append((boxes[i], scores[i], class_ids[i]))
        return results

    def color_callback(self, msg):
        color_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        if self.latest_depth is None:
            return
        
        detections = self.detect(color_frame)
        h, w = self.latest_depth.shape[:2]
        
        for (box, score, class_id) in detections:
            x, y, bw, bh = box
            cx = x + bw // 2
            cy = y + bh // 2
            
            if 0 <= cx < w and 0 <= cy < h:
                distance = self.latest_depth[cy, cx] / 1000.0
            else:
                distance = 0.0
            
            if distance <= 0.0:
                continue
            
            label = CLASSES[class_id] if class_id < len(CLASSES) else str(class_id)
            
            cv2.rectangle(color_frame, (x, y), (x+bw, y+bh), (0,255,0), 2)
            text = f'{label} {score:.2f} | {distance:.2f}m'
            cv2.putText(color_frame, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            self.get_logger().info(f'Detected {label} at {distance:.2f}m')
        
        result_msg = self.bridge.cv2_to_imgmsg(color_frame, encoding='passthrough')
        self.result_pub.publish(result_msg)

def main():
    rclpy.init()
    node = ObjectDetectionDepth()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
