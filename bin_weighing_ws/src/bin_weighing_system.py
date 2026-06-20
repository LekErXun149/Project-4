#!/usr/bin/env python3
"""
Bin Weighing System with Integrated Calibration Check
- Checks calibration age on startup (warns if > 2 hours old)
- Launches calibration wizard if needed
- Manual trigger weighing with sample filtering
- Continuous camera detection
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import queue
from pyzbar.pyzbar import decode, ZBarSymbol
from collections import deque
import sys
import os
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# ── Class names must match order in your trained ONNX model ──────────────────
CLASSES = ['Foam', 'Lipgloss', 'Acid', 'Control', 'Serum', 'Cream']

# ── Path to your trained ONNX model ─────────────────────────────────────────
MODEL_PATH = '/home/group5/best.onnx'

# ── Detection settings ───────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.45
NMS_THRESHOLD         = 0.45
INPUT_SIZE            = 640
DETECT_EVERY_N_FRAMES = 3

# ── Weight sampling settings ─────────────────────────────────────────────────
WEIGHT_SAMPLE_COUNT   = 30    # Number of samples to collect
OUTLIER_STD_THRESHOLD = 2.0   # Remove samples beyond N standard deviations
SAMPLE_RATE_HZ        = 10    # Sample at 10 Hz (100ms interval)

# ── Calibration settings ─────────────────────────────────────────────────────
CALIBRATION_FILE = '/root/.ros/hx711_calibration.json'
CALIBRATION_MAX_AGE_HOURS = 2  # Warn if calibration older than 2 hours

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'


def check_calibration_age():
    """
    Check if calibration file exists and is recent enough
    Returns: (exists, is_recent, age_hours, cal_data)
    """
    if not os.path.exists(CALIBRATION_FILE):
        return False, False, None, None
    
    try:
        # Read calibration file
        with open(CALIBRATION_FILE, 'r') as f:
            cal_data = json.load(f)
        
        # Get file modification time
        file_mtime = os.path.getmtime(CALIBRATION_FILE)
        cal_datetime = datetime.fromtimestamp(file_mtime)
        age = datetime.now() - cal_datetime
        age_hours = age.total_seconds() / 3600
        
        is_recent = age_hours <= CALIBRATION_MAX_AGE_HOURS
        
        return True, is_recent, age_hours, cal_data
        
    except Exception as e:
        print(f"{RED}Error reading calibration file: {e}{RESET}")
        return False, False, None, None


def check_and_warn_calibration():
    """
    Check calibration and print warnings, optionally auto-launch calibration
    Returns: (exists, is_recent, age_hours)
    """
    exists, is_recent, age_hours, cal_data = check_calibration_age()
    
    print(f"\n{CYAN}{'=' * 80}{RESET}")
    print(f"{BOLD}{CYAN}BIN WEIGHING SYSTEM - STARTUP{RESET}")
    print(f"{CYAN}{'=' * 80}{RESET}\n")
    
    if not exists:
        print(f"{RED}✗ CALIBRATION MISSING!{RESET}")
        print(f"{YELLOW}  Location: {CALIBRATION_FILE}{RESET}")
        print(f"{RED}  WARNING: Scale readings will be invalid without calibration!{RESET}\n")
        
        # Wait for other nodes to finish logging
        print(f"{YELLOW}Waiting for system to stabilize...{RESET}")
        import time
        time.sleep(3)
        
        # Prompt to launch calibration wizard - make it very visible
        print(f"\n{CYAN}{'=' * 80}{RESET}")
        print(f"{BOLD}{CYAN}CALIBRATION REQUIRED{RESET}")
        print(f"{CYAN}{'=' * 80}{RESET}")
        sys.stdout.flush()
        
        response = input(f"\n{BOLD}{CYAN}>>> Launch calibration wizard now? (yes/no): {RESET}").strip().lower()
        
        if response in ['yes', 'y']:
            print(f"\n{GREEN}Launching calibration...{RESET}\n")
            launch_calibration_node()
            
            # Re-check after calibration
            exists, is_recent, age_hours, cal_data = check_calibration_age()
            if exists:
                print(f"\n{GREEN}✓ Calibration complete! Proceeding...{RESET}\n")
            else:
                print(f"\n{YELLOW}⚠  Calibration not saved. Continuing anyway...{RESET}\n")
        else:
            print(f"\n{YELLOW}Skipping calibration. Continuing without calibration...{RESET}\n")
        
        return False, False, None
    
    print(f"{GREEN}✓ Calibration file found{RESET}")
    print(f"  Location: {CALIBRATION_FILE}")
    
    if age_hours is not None:
        print(f"  Age: {age_hours:.1f} hours old")
        
        if is_recent:
            print(f"  {GREEN}✓ Recent calibration (within {CALIBRATION_MAX_AGE_HOURS} hours){RESET}")
        else:
            if age_hours < 24:
                print(f"  {YELLOW}⚠  Older than {CALIBRATION_MAX_AGE_HOURS} hours{RESET}")
            else:
                days = age_hours / 24
                print(f"  {YELLOW}⚠  {days:.1f} days old{RESET}")
            
            print(f"  {YELLOW}⚠  Recalibration recommended!{RESET}\n")
            
            # Wait for other nodes to finish logging
            print(f"{YELLOW}Waiting for system to stabilize...{RESET}")
            import time
            time.sleep(3)
            
            # Prompt to launch calibration wizard - make it very visible
            print(f"\n{CYAN}{'=' * 80}{RESET}")
            print(f"{BOLD}{YELLOW}RECALIBRATION RECOMMENDED{RESET}")
            print(f"{CYAN}{'=' * 80}{RESET}")
            sys.stdout.flush()
            
            response = input(f"\n{BOLD}{CYAN}>>> Launch calibration wizard now? (yes/no): {RESET}").strip().lower()
            
            if response in ['yes', 'y']:
                print(f"\n{GREEN}Launching calibration...{RESET}\n")
                launch_calibration_node()
                
                # Re-check after calibration
                exists, is_recent, age_hours, cal_data = check_calibration_age()
                print(f"\n{GREEN}✓ Calibration updated! Proceeding...{RESET}\n")
            else:
                print(f"\n{YELLOW}Using existing calibration. Proceeding...{RESET}\n")
    
    print(f"{GREEN}Starting weighing system...{RESET}\n")
    return exists, is_recent, age_hours


def launch_calibration_node():
    """Launch the calibration wizard node"""
    
    print(f"{CYAN}{'=' * 80}{RESET}")
    print(f"{BOLD}LAUNCHING CALIBRATION WIZARD{RESET}")
    print(f"{CYAN}{'=' * 80}{RESET}\n")
    
    try:
        # Try to run calibration node
        result = subprocess.run([
            'ros2', 'run', 'hx711_scale', 'calibration_node'
        ])
        
        if result.returncode == 0:
            print(f"\n{GREEN}✓ Calibration completed successfully!{RESET}")
            
            # Reload calibration in the running scale_node
            print(f"{CYAN}Reloading calibration in scale_node...{RESET}")
            try:
                reload_result = subprocess.run([
                    'ros2', 'service', 'call', '/reload_calibration', 'std_srvs/srv/Trigger'
                ], capture_output=True, timeout=5)
                
                if reload_result.returncode == 0:
                    print(f"{GREEN}✓ Calibration reloaded in scale_node{RESET}")
                else:
                    print(f"{YELLOW}⚠  Could not reload calibration automatically{RESET}")
                    print(f"{YELLOW}   You may need to restart the system for new calibration to take effect{RESET}")
            except subprocess.TimeoutExpired:
                print(f"{YELLOW}⚠  Calibration reload service timed out{RESET}")
                print(f"{YELLOW}   You may need to restart the system{RESET}")
        else:
            print(f"\n{YELLOW}⚠  Calibration exited with code {result.returncode}{RESET}")
        
    except FileNotFoundError:
        print(f"\n{RED}✗ Could not launch calibration_node{RESET}")
        print(f"{YELLOW}  Command: ros2 run hx711_scale calibration_node{RESET}")
        print(f"{YELLOW}  Make sure hx711_scale package is built and sourced{RESET}")
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Calibration interrupted by user{RESET}")
    except Exception as e:
        print(f"\n{RED}✗ Error launching calibration: {e}{RESET}")


class WeightSampler:
    """Collects weight samples and filters outliers"""
    
    def __init__(self, sample_count=30, outlier_threshold=2.0):
        self.samples = []
        self.sample_count = sample_count
        self.outlier_threshold = outlier_threshold
        
    def add_sample(self, weight):
        self.samples.append(weight)
    
    def is_complete(self):
        return len(self.samples) >= self.sample_count
    
    def get_progress(self):
        return (len(self.samples) / self.sample_count) * 100
    
    def get_filtered_weight(self):
        if len(self.samples) < 5:
            return None, None, []
        
        samples_array = np.array(self.samples)
        mean = np.mean(samples_array)
        std = np.std(samples_array)
        
        if std == 0:
            return mean, 0.0, []
        
        z_scores = np.abs((samples_array - mean) / std)
        mask = z_scores <= self.outlier_threshold
        filtered_samples = samples_array[mask]
        outliers = samples_array[~mask]
        
        if len(filtered_samples) < 5:
            return mean, std, []
        
        filtered_mean = np.mean(filtered_samples)
        filtered_std = np.std(filtered_samples)
        
        return filtered_mean, filtered_std, outliers.tolist()
    
    def clear(self):
        self.samples.clear()


class BinWeighingSystem(Node):
    def __init__(self):
        super().__init__('bin_weighing_system')
        self.bridge = CvBridge()

        # ── Load ONNX model ──────────────────────────────────────────────────
        self.get_logger().info(f'Loading model: {MODEL_PATH}')
        self.net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.get_logger().info('Model loaded successfully')

        # ── State ────────────────────────────────────────────────────────────
        self.latest_depth      = None
        self.latest_weight     = 0.0
        self.latest_barcode    = 'N/A'
        self.latest_detections = []
        self.detected_items_cache = []
        self.frame_count       = 0
        
        # ── Sampling state ───────────────────────────────────────────────────
        self.sampling_active   = False
        self.sampler          = WeightSampler(WEIGHT_SAMPLE_COUNT, OUTLIER_STD_THRESHOLD)
        self.final_weight     = 0.0
        self.final_weight_std = 0.0
        self.final_barcode    = 'N/A'
        self.final_items      = []

        # ── Detection thread ─────────────────────────────────────────────────
        self.frame_queue  = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.detect_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detect_thread.start()
        
        # ── User input thread ────────────────────────────────────────────────
        self.input_queue = queue.Queue()
        self.input_thread = threading.Thread(target=self.input_loop, daemon=True)
        self.input_thread.start()
        
        # ── Sampling timer ───────────────────────────────────────────────────
        self.sample_timer = self.create_timer(1.0 / SAMPLE_RATE_HZ, self.sample_timer_callback)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.color_sub = self.create_subscription(Image, '/camera0/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera0/depth/image_raw', self.depth_callback, 10)
        self.picam_sub = self.create_subscription(Image, '/camera_node/image_raw', self.barcode_callback, 10)
        self.weight_sub = self.create_subscription(Float64, '/weight', self.weight_callback, 10)

        # ── Publishers ───────────────────────────────────────────────────────
        self.detection_pub     = self.create_publisher(Image,  '/detection/image',     10)
        self.barcode_debug_pub = self.create_publisher(Image,  '/barcode/debug_image', 10)
        self.barcode_pub       = self.create_publisher(String, '/barcode/data',        10)
        self.result_pub        = self.create_publisher(String, '/bin/result',          10)

        self.get_logger().info('=' * 80)
        self.get_logger().info('Bin Weighing System started (Manual Trigger Mode)')
        self.get_logger().info('=' * 80)
        self.get_logger().info(f'Weight sampling: {WEIGHT_SAMPLE_COUNT} samples at {SAMPLE_RATE_HZ} Hz')
        self.get_logger().info('HOW TO USE:')
        self.get_logger().info('  1. Cameras run continuously')
        self.get_logger().info('  2. Place item on scale')
        self.get_logger().info('  3. Press ENTER to start weight sampling')
        self.get_logger().info('  4. Wait ~3 seconds for sampling')
        self.get_logger().info('  5. Results displayed')
        self.get_logger().info('=' * 80)
        print('\n>>> Press ENTER to weigh item... ', end='', flush=True)

    def input_loop(self):
        while True:
            try:
                input()
                self.input_queue.put('trigger')
            except:
                break

    def weight_callback(self, msg):
        self.latest_weight = msg.data

    def sample_timer_callback(self):
        if not self.input_queue.empty():
            self.input_queue.get()
            if not self.sampling_active:
                self.start_sampling()
        
        if self.sampling_active:
            self.sampler.add_sample(self.latest_weight)
            progress = self.sampler.get_progress()
            
            if int(progress) % 10 == 0 and len(self.sampler.samples) > 0:
                self.get_logger().info(f'Sampling... {progress:.0f}% ({len(self.sampler.samples)}/{WEIGHT_SAMPLE_COUNT})')
            
            if self.sampler.is_complete():
                self.finish_sampling()

    def start_sampling(self):
        self.sampling_active = True
        self.sampler.clear()
        self.get_logger().info('=' * 80)
        self.get_logger().info('Starting weight sampling...')
        self.get_logger().info(f'Collecting {WEIGHT_SAMPLE_COUNT} samples at {SAMPLE_RATE_HZ} Hz')
        
        self.detected_items_cache = self.latest_detections.copy()
        self.final_barcode = self.latest_barcode

    def finish_sampling(self):
        self.sampling_active = False
        
        weight, std, outliers = self.sampler.get_filtered_weight()
        
        if weight is None:
            self.get_logger().error('Not enough valid samples!')
            print('\n>>> Press ENTER to weigh item... ', end='', flush=True)
            return
        
        self.final_weight = weight
        self.final_weight_std = std
        
        self.final_items = []
        for (box, score, class_id) in self.detected_items_cache:
            x, y, bw, bh = box
            cx = x + bw // 2
            cy = y + bh // 2
            distance = self.get_depth_at(cx, cy)
            
            if distance > 0.0:
                label = CLASSES[class_id] if class_id < len(CLASSES) else str(class_id)
                self.final_items.append(f'{label}({distance:.2f}m)')
        
        # ── Output results ────────────────────────────────────────────────────
        self.get_logger().info('-' * 80)
        self.get_logger().info('SAMPLING COMPLETE')
        self.get_logger().info(f'  Samples collected: {len(self.sampler.samples)}')
        self.get_logger().info(f'  Outliers removed: {len(outliers)} samples {outliers}')
        self.get_logger().info('-' * 80)
        self.get_logger().info('RESULTS:')
        self.get_logger().info(f'  Items detected : {", ".join(self.final_items) if self.final_items else "None"}')
        self.get_logger().info(f'  Weight        : {self.final_weight:.2f}g (±{self.final_weight_std:.2f}g)')
        self.get_logger().info(f'  Barcode       : {self.final_barcode}')
        self.get_logger().info('=' * 80)
        
        if self.final_items or self.final_barcode != 'N/A':
            result = String()
            result.data = (
                f'Items: {", ".join(self.final_items) if self.final_items else "None"} | '
                f'Weight: {self.final_weight:.2f}g | '
                f'Barcode: {self.final_barcode}'
            )
            self.result_pub.publish(result)
        
        print('\n>>> Press ENTER to weigh next item... ', end='', flush=True)

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'Depth conversion error: {e}')

    def barcode_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'Barcode image conversion error: {e}')
            return

        barcodes = decode(frame, symbols=[ZBarSymbol.CODE39, ZBarSymbol.CODE128])
        for barcode in barcodes:
            data  = barcode.data.decode('utf-8')
            btype = barcode.type

            if data != self.latest_barcode:
                self.latest_barcode = data
                self.get_logger().info(f'Barcode detected {btype}: {data}')
                barcode_msg      = String()
                barcode_msg.data = f'{btype}:{data}'
                self.barcode_pub.publish(barcode_msg)

            pts = barcode.polygon
            if pts:
                pts_list = [(p.x, p.y) for p in pts]
                for i in range(len(pts_list)):
                    cv2.line(frame, pts_list[i], pts_list[(i+1) % len(pts_list)], (0, 255, 0), 2)
            cv2.putText(frame, f'{btype}: {data}',
                       (barcode.rect.left, max(barcode.rect.top - 10, 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        try:
            debug_msg        = self.bridge.cv2_to_imgmsg(frame, encoding='passthrough')
            debug_msg.header = msg.header
            self.barcode_debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().warn(f'Barcode debug publish error: {e}')

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        outputs = np.array(outputs[0]).T

        boxes, scores, class_ids = [], [], []
        for row in outputs:
            classes_score = row[4:]
            max_score     = np.max(classes_score)
            if max_score >= CONFIDENCE_THRESHOLD:
                class_id = np.argmax(classes_score)
                cx, cy, bw, bh = row[:4]
                x      = int((cx - bw/2) * w / INPUT_SIZE)
                y      = int((cy - bh/2) * h / INPUT_SIZE)
                width  = int(bw * w / INPUT_SIZE)
                height = int(bh * h / INPUT_SIZE)
                boxes.append([x, y, width, height])
                scores.append(float(max_score))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        results = []
        for i in indices:
            results.append((boxes[i], scores[i], class_ids[i]))
        return results

    def detection_loop(self):
        while True:
            frame      = self.frame_queue.get()
            detections = self.detect(frame)
            if not self.result_queue.full():
                self.result_queue.put(detections)

    def get_depth_at(self, cx, cy):
        if self.latest_depth is None:
            return 0.0
        h, w = self.latest_depth.shape[:2]
        if 0 <= cx < w and 0 <= cy < h:
            region = self.latest_depth[max(0, cy-5):min(h, cy+5), max(0, cx-5):min(w, cx+5)]
            valid = region[region > 0]
            return float(np.mean(valid)) / 1000.0 if len(valid) > 0 else 0.0
        return 0.0

    def color_callback(self, msg):
        self.frame_count += 1

        try:
            color_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'Color image conversion error: {e}')
            return

        if self.frame_count % DETECT_EVERY_N_FRAMES == 0:
            if not self.frame_queue.full():
                self.frame_queue.put(color_frame.copy())

        if not self.result_queue.empty():
            self.latest_detections = self.result_queue.get()

        detected_items = []
        for (box, score, class_id) in self.latest_detections:
            x, y, bw, bh = box
            cx = x + bw // 2
            cy = y + bh // 2
            distance = self.get_depth_at(cx, cy)

            if distance <= 0.0:
                continue

            label = CLASSES[class_id] if class_id < len(CLASSES) else str(class_id)
            detected_items.append(f'{label}({distance:.2f}m)')

            cv2.rectangle(color_frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            text = f'{label} {score:.2f} | {distance:.2f}m'
            cv2.putText(color_frame, text, (x, max(y-10, 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.sampling_active:
            progress = self.sampler.get_progress()
            status_text = f"SAMPLING: {progress:.0f}%"
            status_color = (0, 165, 255)
        else:
            status_text = "READY - Press ENTER"
            status_color = (0, 255, 0)
        
        overlay_items = [
            (f'Weight  : {self.latest_weight:.2f}g', (10, 30),  (0, 0, 255)),
            (f'Barcode : {self.latest_barcode}',     (10, 60),  (255, 0, 0)),
            (f'Items   : {len(detected_items)}',     (10, 90),  (0, 255, 255)),
            (f'Status  : {status_text}',             (10, 120), status_color),
        ]
        for text, pos, color in overlay_items:
            cv2.putText(color_frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        try:
            result_msg = self.bridge.cv2_to_imgmsg(color_frame, encoding='passthrough')
            self.detection_pub.publish(result_msg)
        except Exception as e:
            self.get_logger().warn(f'Detection publish error: {e}')


def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bin Weighing System')
    parser.add_argument('--skip-cal-check', action='store_true',
                       help='Skip calibration age check on startup')
    args, unknown = parser.parse_known_args()
    
    # Initialize ROS2 FIRST
    rclpy.init()
    
    # THEN check calibration (after ROS2 is initialized)
    if not args.skip_cal_check:
        check_and_warn_calibration()
    else:
        print(f"\n{YELLOW}Skipping calibration check...{RESET}\n")
    
    # Start weighing system node
    node = BinWeighingSystem()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Shutting down...{RESET}\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
