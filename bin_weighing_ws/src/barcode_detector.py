import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from pyzbar.pyzbar import decode
 
class BarcodeDetector(Node):
    def __init__(self):
        super().__init__('barcode_detector')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        # Publisher for annotated debug image (view with: ros2 run rqt_image_view rqt_image_view)
        self.debug_pub = self.create_publisher(Image, '/barcode_detector/debug_image', 10)
        # Publisher for detected barcode data
        self.barcode_pub = self.create_publisher(String, '/barcode_detector/barcode_data', 10)
 
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        barcodes = decode(frame)
        for barcode in barcodes:
            data = barcode.data.decode('utf-8')
            btype = barcode.type
            self.get_logger().info(f'Detected {btype}: {data}')
 
            # Publish barcode string data
            barcode_msg = String()
            barcode_msg.data = f'{btype}:{data}'
            self.barcode_pub.publish(barcode_msg)
 
            # Draw rectangle around barcode
            pts = barcode.polygon
            if pts:
                pts = [(p.x, p.y) for p in pts]
                for i in range(len(pts)):
                    cv2.line(frame, pts[i], pts[(i+1) % len(pts)], (0, 255, 0), 2)
            cv2.putText(frame, data, (barcode.rect.left, barcode.rect.top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
        # Publish annotated frame instead of imshow (works headless / over SSH)
        debug_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        debug_msg.header = msg.header  # preserve timestamp & frame_id
        self.debug_pub.publish(debug_msg)
 
 
def main():
    rclpy.init()
    node = BarcodeDetector()
    rclpy.spin(node)
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()
