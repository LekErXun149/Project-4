#!/usr/bin/env python3
"""
Bin Weighing Backend (Program 2 - ROS2 Backend)
Runs in background, handles all ROS2 operations
Communicates with manager via topics
All logging happens here (hidden from user)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32, Float64, Empty
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import queue
from pyzbar.pyzbar import decode, ZBarSymbol
from collections import deque
import pandas as pd
import os

# ── Class names (match your YOLO model) ──────────────────────────────────────
CLASSES = ['Foam', 'Lipgloss', 'Acid', 'Control', 'Serum', 'Cream']

# ── Model path ───────────────────────────────────────────────────────────────
MODEL_PATH = '/home/group5/best.onnx'

# ── Detection settings ───────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.45
NMS_THRESHOLD         = 0.45
INPUT_SIZE            = 640      # YOLO input size
DETECT_EVERY_N_FRAMES = 60       # Detect every 60th frame for low CPU usage

# ── Weight sampling settings ─────────────────────────────────────────────────
WEIGHT_SAMPLE_COUNT   = 70     # Higher value for better averaging after 3s stabilization
OUTLIER_STD_THRESHOLD = 2.0    # MAD threshold (2 * mad, same as calibration)
SAMPLE_RATE_HZ        = 10

# ── Bin database settings ────────────────────────────────────────────────────
BIN_DATABASE_CSV = '/home/group5/bin_weighing_ws/src/bins_master.csv'  # Path to bin database
WEIGHT_TOLERANCE_PERCENT = 2.0  # ±2% tolerance for weight matching

# Expected CSV format (multiple rows per bin for multi-SKU support):
# bin_barcode,compartment,bin_tare_weight,item_name,quantity,unit_weight
# BIN001,1,500,Acid,5,20
# BIN001,2,500,Serum,3,25
# BIN001,3,500,,,
# BIN002,1,500,Lipgloss,8,18
# Note: bin_tare_weight should be same for all rows of same bin


class WeightSampler:
    """Collects and filters weight samples"""
    
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
        
        # Use MAD (Median Absolute Deviation) like calibration wizard
        median = np.median(samples_array)
        mad = np.median(np.abs(samples_array - median))
        
        if mad == 0:
            # All samples identical
            return median, 0.0, []
        
        # Filter using MAD (more robust than z-score)
        threshold = self.outlier_threshold * mad
        mask = np.abs(samples_array - median) < threshold
        filtered_samples = samples_array[mask]
        outliers = samples_array[~mask]
        
        if len(filtered_samples) < 5:
            # Too many outliers, return unfiltered
            return np.mean(samples_array), np.std(samples_array), []
        
        filtered_mean = np.mean(filtered_samples)
        filtered_std = np.std(filtered_samples)
        
        return filtered_mean, filtered_std, outliers.tolist()
    
    def clear(self):
        self.samples.clear()


class BinWeighingBackend(Node):
    """Backend node - handles all ROS2 operations"""
    
    def __init__(self):
        super().__init__('bin_weighing_backend')
        self.bridge = CvBridge()

        # ── Load ONNX model ──────────────────────────────────────────────────
        self.get_logger().info(f'Loading model: {MODEL_PATH}')
        self.net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.get_logger().info('Model loaded')
        
        # ── Load bin database ────────────────────────────────────────────────
        self.bin_database = self.load_bin_database()
        if self.bin_database is not None:
            self.get_logger().info(f'Loaded {len(self.bin_database)} bins from database')
        else:
            self.get_logger().warn('Bin database not loaded - weight verification disabled')

        # ── State ────────────────────────────────────────────────────────────
        self.latest_depth      = None
        self.latest_weight_raw = 0.0  # Store RAW ADC value from scale_node
        self.latest_barcode    = 'N/A'
        self.latest_detections = []
        self.frame_count       = 0
        
        # ── Sampling state ───────────────────────────────────────────────────
        self.sampling_active   = False
        self.sampler          = WeightSampler(WEIGHT_SAMPLE_COUNT, OUTLIER_STD_THRESHOLD)
        self.measurement_id   = 0  # Track each measurement uniquely
        self.measurement_stage = 'ready'  # 'ready' → 'taring' → 'tared' → 'stabilizing' → 'measuring' → 'ready'
        self.tare_reading = None  # Store tare reading from stage 1
        self.tare_samples = []  # Collect tare samples
        self.tare_sample_count = 20  # Reduced to 20 to handle slower publishing rates (~5 Hz)
        self.tare_retry_count = 0  # Track number of tare retries
        self.max_tare_retries = 2  # Maximum automatic retries before giving up

        # ── Detection thread ─────────────────────────────────────────────────
        self.frame_queue  = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.detect_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detect_thread.start()
        
        # ── Sampling timer ───────────────────────────────────────────────────
        self.sample_timer = self.create_timer(1.0 / SAMPLE_RATE_HZ, self.sample_timer_callback)
        
        # ── Stabilization check timer ────────────────────────────────────────
        # Checks if 2-second stabilization period is complete
        self.stabilization_timer = self.create_timer(0.1, self.check_stabilization)
        self.stabilization_start_time = None

        # ── Subscribers ──────────────────────────────────────────────────────
        self.color_sub = self.create_subscription(Image, '/camera0/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera0/depth/image_raw', self.depth_callback, 10)
        self.picam_sub = self.create_subscription(Image, '/camera_node/image_raw', self.barcode_callback, 10)
        
        # Subscribe to RAW weight (ADC values) for better filtering accuracy
        # scale_node must publish to /weight/raw topic
        self.weight_raw_sub = self.create_subscription(Float64, '/weight/raw', self.weight_raw_callback, 10)
        
        # Subscribe to trigger from manager
        self.trigger_sub = self.create_subscription(Empty, '/start_weighing', self.trigger_callback, 10)
        
        # Create a timer to check if we're receiving weight data
        self.create_timer(2.0, self.check_weight_data_once)
        self.weight_data_checked = False

        # ── Publishers ───────────────────────────────────────────────────────
        self.detection_pub     = self.create_publisher(Image,  '/detection/image',     10)
        self.barcode_debug_pub = self.create_publisher(Image,  '/barcode/debug_image', 10)
        self.barcode_pub       = self.create_publisher(String, '/barcode/data',        10)
        
        # Publishers for manager
        self.progress_pub = self.create_publisher(Float32, '/weighing_progress', 10)
        self.result_pub   = self.create_publisher(String,  '/weighing_result',   10)
        self.items_pub    = self.create_publisher(String,  '/current_items',     10)
        self.debug_pub    = self.create_publisher(String,  '/debug_sampling',    10)  # Debug messages

        self.get_logger().info('Backend ready')

    def trigger_callback(self, msg):
        """
        Handle trigger for 2-stage measurement:
        Stage 1 (ready): Platform empty → Take tare reading
        Stage 2 (tared): Item on platform → Measure weight
        """
        if not self.sampling_active:
            if self.measurement_stage == 'ready':
                # Stage 1: Take tare reading
                self.get_logger().info('▶ Stage 1: Taking tare reading')
                self.take_tare()
            elif self.measurement_stage == 'tared':
                # Stage 2: Measure weight
                self.get_logger().info('▶ Stage 2: Measuring weight')
                self.start_sampling()
        else:
            self.get_logger().warn('⚠ Trigger ignored - sampling already in progress')
    
    def take_tare(self):
        """Stage 1: Start tare collection (non-blocking with timeout)"""
        self.get_logger().info('  Collecting tare samples (platform should be empty)...')
        self.tare_samples = []
        self.measurement_stage = 'taring'
        self.tare_start_time = self.get_clock().now()  # Track when taring started
    
    def finish_tare(self):
        """Process tare samples with outlier filtering"""
        import statistics
        
        if not self.tare_samples:
            self.get_logger().error('  ✗ Tare failed - no samples collected!')
            self.get_logger().error('  Check if /weight/raw topic is publishing')
            self.measurement_stage = 'ready'
            return
        
        # STEP 1: Physical plausibility filter on tare samples
        # Empty platform should be between -100g and +100g (with platform mass ~500g, reading should be 450-550g range)
        # But we're more lenient: -100g to 1000g to account for platform variations
        physically_valid_tare = []
        physics_outliers_tare = []
        
        for s in self.tare_samples:
            converted = self.raw_to_weight(s)
            # Empty platform shouldn't be negative or extremely heavy
            if -100.0 <= converted <= 1000.0:
                physically_valid_tare.append(s)
            else:
                physics_outliers_tare.append(s)
        
        # Check if we have enough valid samples after physics filter
        if len(physically_valid_tare) < 10:
            self.get_logger().error(f'  ✗ Tare failed - only {len(physically_valid_tare)}/20 samples were physically valid')
            self.get_logger().error(f'  {len(physics_outliers_tare)} samples were load cell glitches')
            self.get_logger().error('  Platform may be unstable or load cell needs calibration')
            self.measurement_stage = 'ready'
            self.tare_samples = []
            
            # Notify manager of failure
            error_msg = String()
            error_msg.data = "TARE_FAILED"
            self.result_pub.publish(error_msg)
            return
        
        # STEP 2: Apply MAD outlier filtering to physically valid tare samples
        median = statistics.median(physically_valid_tare)
        mad = statistics.median([abs(s - median) for s in physically_valid_tare])
        threshold = OUTLIER_STD_THRESHOLD * mad
        
        # Filter outliers
        mad_outliers_tare = []
        valid_tare = []
        for s in physically_valid_tare:
            if mad > 0 and abs(s - median) > threshold:
                mad_outliers_tare.append(s)
            else:
                valid_tare.append(s)
        
        # Final check: need at least 10 valid samples after both filters
        if len(valid_tare) < 10:
            self.get_logger().error(f'  ✗ Tare failed - only {len(valid_tare)} valid samples after filtering')
            self.get_logger().error('  Platform too unstable - try again')
            self.measurement_stage = 'ready'
            self.tare_samples = []
            
            # Notify manager of failure
            error_msg = String()
            error_msg.data = "TARE_FAILED"
            self.result_pub.publish(error_msg)
            return
        
        # Use filtered average
        self.tare_reading = statistics.mean(valid_tare)
        std = statistics.stdev(valid_tare) if len(valid_tare) > 1 else 0
        
        total_tare_outliers = len(physics_outliers_tare) + len(mad_outliers_tare)
        
        self.get_logger().info(f'  ✓ Tare complete: {self.tare_reading:.0f} ADC (±{std:.0f})')
        if total_tare_outliers > 0:
            self.get_logger().info(f'  Removed {total_tare_outliers} outlier samples ({len(physics_outliers_tare)} physics, {len(mad_outliers_tare)} MAD)')
        
        # Check tare quality and warn user if poor
        outlier_percentage = (total_tare_outliers / len(self.tare_samples)) * 100
        quality_warning = False
        
        if outlier_percentage > 30:  # More than 30% outliers
            self.get_logger().warn('=' * 70)
            self.get_logger().warn('⚠️  TARE QUALITY WARNING')
            self.get_logger().warn('=' * 70)
            self.get_logger().warn(f'Removed {total_tare_outliers}/20 samples ({outlier_percentage:.0f}% outliers)')
            self.get_logger().warn('Platform appears unstable - tare quality is POOR')
            self.get_logger().warn('This will affect measurement accuracy!')
            self.get_logger().warn('=' * 70)
            quality_warning = True
        elif std > 500:  # High standard deviation
            self.get_logger().warn('=' * 70)
            self.get_logger().warn('⚠️  TARE QUALITY WARNING')
            self.get_logger().warn('=' * 70)
            self.get_logger().warn(f'Standard deviation: ±{std:.0f} ADC (high instability)')
            self.get_logger().warn('Platform readings are not stable')
            self.get_logger().warn('This will affect measurement accuracy!')
            self.get_logger().warn('=' * 70)
            quality_warning = True
        
        # GENERATE DEBUG OUTPUT FOR STAGE 1 (TARE)
        debug_lines = []
        debug_lines.append(f'DEBUG: STAGE 1 - TARE')
        debug_lines.append(f'(Two-stage filtering: 1) Physics check, 2) MAD outlier removal)')
        debug_lines.append('')
        
        debug_lines.append(f'Total tare samples collected: {len(self.tare_samples)}')
        debug_lines.append('')
        
        # Show all tare samples with markers
        debug_lines.append('ALL tare samples (RAW ADC):')
        for i, s in enumerate(self.tare_samples):
            converted = self.raw_to_weight(s)
            is_physics_outlier = s in physics_outliers_tare
            is_mad_outlier = s in mad_outliers_tare
            if is_physics_outlier:
                marker = '[PHYSICS OUTLIER]'
            elif is_mad_outlier:
                marker = '[MAD OUTLIER]'
            else:
                marker = '✓'
            debug_lines.append(f'  {marker:20s} Sample {i+1:2d}: RAW={s:.0f}  →  {converted:8.2f}g')
        
        # Statistics (on ALL samples before filtering)
        debug_lines.append('')
        debug_lines.append('Tare statistics (before filtering):')
        debug_lines.append(f'  Mean: {statistics.mean(self.tare_samples):.0f} ADC')
        debug_lines.append(f'  Median: {statistics.median(self.tare_samples):.0f} ADC')
        debug_lines.append(f'  Std Dev: {statistics.stdev(self.tare_samples) if len(self.tare_samples) > 1 else 0:.0f} ADC')
        debug_lines.append(f'  Min: {min(self.tare_samples):.0f} ADC')
        debug_lines.append(f'  Max: {max(self.tare_samples):.0f} ADC')
        debug_lines.append(f'  Range: {max(self.tare_samples) - min(self.tare_samples):.0f} ADC')
        
        # Physics filtering info
        debug_lines.append('')
        debug_lines.append('STEP 1: Physical Plausibility Filter (Tare)')
        debug_lines.append(f'  Valid range: -100g to 1000g (empty platform with variations)')
        debug_lines.append(f'  Physics outliers removed: {len(physics_outliers_tare)}')
        debug_lines.append(f'  Physically valid samples: {len(physically_valid_tare)}/{len(self.tare_samples)}')
        
        # MAD filtering info on physically valid samples
        debug_lines.append('')
        debug_lines.append('STEP 2: MAD Outlier Detection (on physically valid tare samples)')
        debug_lines.append(f'  Median: {median:.0f} ADC')
        debug_lines.append(f'  MAD: {mad:.0f} ADC')
        debug_lines.append(f'  Threshold: ±{threshold:.0f} ADC')
        debug_lines.append(f'  MAD outliers removed: {len(mad_outliers_tare)}')
        
        # Valid samples used
        debug_lines.append('')
        debug_lines.append(f'Final valid tare samples (after both filters - {len(valid_tare)} total):')
        for i, s in enumerate(valid_tare[:10]):  # Show first 10
            converted = self.raw_to_weight(s)
            debug_lines.append(f'  ✓ Valid {i+1:2d}: RAW={s:.0f}  →  {converted:.2f}g')
        if len(valid_tare) > 10:
            debug_lines.append(f'  ... and {len(valid_tare) - 10} more valid samples')
        
        # Summary
        debug_lines.append('')
        debug_lines.append('=' * 80)
        debug_lines.append('TARE SUMMARY:')
        debug_lines.append('=' * 80)
        debug_lines.append(f'Tare reading:  {self.tare_reading:.0f} ADC  →  {self.raw_to_weight(self.tare_reading):.2f}g')
        debug_lines.append(f'Stability:     ±{std:.0f} ADC')
        debug_lines.append(f'Physics outliers: {len(physics_outliers_tare)} (load cell glitches)')
        debug_lines.append(f'MAD outliers:     {len(mad_outliers_tare)} (statistical outliers)')
        debug_lines.append(f'Total outliers:   {total_tare_outliers}/{len(self.tare_samples)} removed')
        debug_lines.append(f'Valid samples: {len(valid_tare)}/{len(self.tare_samples)} (minimum 10 required)')
        if std < 100:
            debug_lines.append(f'Quality:       ✓ EXCELLENT (Std Dev < 100)')
        elif std < 500:
            debug_lines.append(f'Quality:       ✓ GOOD (Std Dev < 500)')
        elif std < 1000:
            debug_lines.append(f'Quality:       ⚠ FAIR (Std Dev < 1000)')
        else:
            debug_lines.append(f'Quality:       ✗ POOR (Std Dev > 1000) - Check platform stability')
        debug_lines.append('=' * 80)
        
        # Publish debug data for Stage 1
        debug_msg = String()
        debug_msg.data = '\n'.join(debug_lines)
        self.debug_pub.publish(debug_msg)
        
        # Small delay to ensure debug message is sent
        import time
        time.sleep(0.05)
        
        # Move to stage 2
        self.measurement_stage = 'tared'
        self.tare_retry_count = 0  # Reset retry counter on success
        self.get_logger().info('  Ready for Stage 2: Place item and press ENTER')
        
        # Publish a simple result to unblock the manager
        # (manager is waiting for a result after pressing ENTER)
        result_msg = String()
        result_msg.data = "TARE_COMPLETE"
        self.result_pub.publish(result_msg)
        
        # Small delay to ensure message is transmitted
        import time
        time.sleep(0.1)
    
    def start_sampling(self):
        """Stage 2: Start weight sampling (after 3-second stabilization)"""
        self.get_logger().info('  Waiting 3 seconds for platform to stabilize...')
        
        # Set state to measuring with delay
        self.measurement_stage = 'stabilizing'
        self.stabilization_start_time = self.get_clock().now()
    
    def check_stabilization(self):
        """Check if 3-second stabilization period is complete"""
        if self.measurement_stage == 'stabilizing':
            elapsed = (self.get_clock().now() - self.stabilization_start_time).nanoseconds / 1e9
            
            if elapsed >= 3.0:  # 3 seconds for better balance of speed and stability
                # Stabilization complete, start actual sampling
                self.sampler.clear()
                self.measurement_id += 1
                self.sampling_active = True
                self.measurement_stage = 'measuring'
                
                self.get_logger().info(f'✓ Sampling started - Measurement #{self.measurement_id}')

    def load_bin_database(self):
        """Load bin database from CSV"""
        try:
            if not os.path.exists(BIN_DATABASE_CSV):
                self.get_logger().error(f'Bin database not found: {BIN_DATABASE_CSV}')
                return None
            
            df = pd.read_csv(BIN_DATABASE_CSV)
            self.get_logger().info(f'Loaded bin database with columns: {list(df.columns)}')
            return df
        except Exception as e:
            self.get_logger().error(f'Failed to load bin database: {e}')
            return None
    
    def raw_to_weight(self, raw_value):
        """
        Convert raw ADC value to weight using calibration from file
        
        Args:
            raw_value: Raw ADC reading
        
        Returns:
            Weight in grams
        """
        try:
            import json
            from pathlib import Path
            
            # Load calibration file (same path as scale_node uses)
            cal_file = Path.home() / '.ros' / 'hx711_calibration.json'
            
            if not cal_file.exists():
                self.get_logger().warn('No calibration file - using default conversion')
                # Default fallback (from interpolation_logic.py)
                offset = -217106
                scale = -112.81  # Average of default points
                return (raw_value - offset) / scale
            
            with open(cal_file, 'r') as f:
                cal_data = json.load(f)
            
            # Get active calibration
            active_id = cal_data.get('active_calibration')
            calibrations = cal_data.get('calibrations', [])
            
            if not calibrations:
                self.get_logger().error('No calibrations in file!')
                return raw_value  # Return raw if no calibration
            
            # Find the active calibration in the list
            cal = None
            if active_id:
                for c in calibrations:
                    if c.get('id') == active_id:
                        cal = c
                        break
            
            # If no active calibration found, use the first one
            if cal is None:
                self.get_logger().warn('No active calibration - using first one')
                cal = calibrations[0]
            
            offset = cal['offset']
            cal_type = cal.get('type', 'single_point')
            
            if cal_type == 'multi_point':
                # Use multi-point interpolation
                points = cal['points']
                # Simple iterative conversion for multi-point
                weight_estimate = (raw_value - offset) / points[0]['scale']
                for _ in range(3):  # 3 iterations
                    # Find two closest calibration points
                    points_sorted = sorted(points, key=lambda p: p['weight'])
                    
                    # Find bracket
                    lower = points_sorted[0]
                    upper = points_sorted[-1]
                    for i in range(len(points_sorted) - 1):
                        if points_sorted[i]['weight'] <= weight_estimate <= points_sorted[i+1]['weight']:
                            lower = points_sorted[i]
                            upper = points_sorted[i+1]
                            break
                    
                    # Interpolate
                    if lower['weight'] == upper['weight']:
                        scale_interp = lower['scale']
                    else:
                        weight_frac = (weight_estimate - lower['weight']) / (upper['weight'] - lower['weight'])
                        scale_interp = lower['scale'] + weight_frac * (upper['scale'] - lower['scale'])
                    
                    weight_estimate = (raw_value - offset) / scale_interp
                
                return weight_estimate
            else:
                # Single-point calibration
                scale = cal['points'][0]['scale']
                return (raw_value - offset) / scale
        
        except Exception as e:
            self.get_logger().error(f'Error converting raw to weight: {e}')
            return raw_value  # Return raw on error
    
    def verify_weight(self, barcode, measured_weight):
        """
        Verify if measured weight matches expected weight from database
        Handles multiple SKUs per bin (up to 8 compartments)
        
        Args:
            barcode: Bin barcode (e.g., 'CODE39:BIN001' or 'BIN001')
            measured_weight: Weight measured by scale (grams)
        
        Returns:
            tuple: (status, expected_weight, tolerance, message)
                   status: 'MATCH', 'MISMATCH', 'NO_DATA', or 'EMPTY_BIN'
        """
        if self.bin_database is None:
            return ('NO_DATA', 0, 0, 'Database not loaded')
        
        # Extract barcode value (remove CODE39: prefix if present)
        if ':' in barcode:
            barcode = barcode.split(':')[1]
        
        # Convert to uppercase for case-insensitive matching
        barcode = barcode.upper()
        
        # Get ALL rows for this bin (multiple SKUs) - case insensitive
        bin_rows = self.bin_database[self.bin_database['bin_barcode'].str.upper() == barcode]
        
        if bin_rows.empty:
            return ('NO_DATA', 0, 0, f'Bin {barcode} not found in database')
        
        # Get bin tare weight (should be same across all rows)
        bin_tare = bin_rows.iloc[0]['bin_tare_weight']
        
        # Calculate total items weight from all SKUs
        total_items_weight = 0
        sku_list = []
        has_items = False
        
        for _, row in bin_rows.iterrows():
            # Check if this row has item data (not empty compartment)
            if pd.notna(row['item_name']) and pd.notna(row['quantity']) and pd.notna(row['unit_weight']):
                has_items = True
                item_weight = row['quantity'] * row['unit_weight']
                total_items_weight += item_weight
                sku_list.append(f"{int(row['quantity'])}× {row['item_name']}")
        
        # Calculate expected total weight
        expected_weight = bin_tare + total_items_weight
        tolerance = expected_weight * (WEIGHT_TOLERANCE_PERCENT / 100.0)
        
        # Check if within tolerance
        weight_diff = abs(measured_weight - expected_weight)
        
        if not has_items:
            # Empty bin - no items configured
            if weight_diff <= tolerance:
                return ('MATCH', expected_weight, tolerance, 'Empty bin weight OK')
            else:
                return ('MISMATCH', expected_weight, tolerance, 'Empty bin weight incorrect')
        
        # Bin has items - verify total weight
        if weight_diff <= tolerance:
            status = 'MATCH'
            message = ', '.join(sku_list)
        else:
            status = 'MISMATCH'
            percent_diff = (measured_weight - expected_weight) / expected_weight * 100
            message = f'Expected {", ".join(sku_list)}, off by {percent_diff:+.1f}%'
        
        return (status, expected_weight, tolerance, message)

    def weight_raw_callback(self, msg):
        """Store RAW ADC value from scale_node"""
        self.latest_weight_raw = msg.data
    
    def check_weight_data_once(self):
        """Check if we're receiving weight data (runs once after 2 seconds)"""
        if self.weight_data_checked:
            return
        
        self.weight_data_checked = True
        
        if self.latest_weight_raw == 0.0:
            self.get_logger().error('=' * 70)
            self.get_logger().error('⚠️  WARNING: NOT RECEIVING /weight/raw DATA!')
            self.get_logger().error('=' * 70)
            self.get_logger().error('The backend is not receiving data from /weight/raw topic.')
            self.get_logger().error('')
            self.get_logger().error('This will cause the system to hang during measurements!')
            self.get_logger().error('')
            self.get_logger().error('TROUBLESHOOTING:')
            self.get_logger().error('  1. Check if scale_node is running:')
            self.get_logger().error('     ros2 node list | grep scale')
            self.get_logger().error('')
            self.get_logger().error('  2. Check if /weight/raw topic exists:')
            self.get_logger().error('     ros2 topic list | grep weight')
            self.get_logger().error('')
            self.get_logger().error('  3. Check if topic is publishing:')
            self.get_logger().error('     ros2 topic echo /weight/raw')
            self.get_logger().error('')
            self.get_logger().error('  4. Verify scale_node.py has RAW publishing')
            self.get_logger().error('=' * 70)
        else:
            self.get_logger().info(f'✓ Receiving /weight/raw data (current: {self.latest_weight_raw:.0f} ADC)')

    def sample_timer_callback(self):
        """Collect samples for both taring and measuring"""
        if self.measurement_stage == 'taring':
            # Check for timeout (8 seconds max for taring - allows for slower publishing rates)
            if hasattr(self, 'tare_start_time'):
                elapsed = (self.get_clock().now() - self.tare_start_time).nanoseconds / 1e9
                if elapsed > 8.0:
                    self.get_logger().error(f'  ✗ Tare timeout after {elapsed:.1f}s!')
                    self.get_logger().error(f'  Only collected {len(self.tare_samples)} samples')
                    
                    # Check if we can retry
                    if self.tare_retry_count < self.max_tare_retries:
                        self.tare_retry_count += 1
                        self.get_logger().warn(f'  ⟳ Retrying tare (attempt {self.tare_retry_count + 1}/{self.max_tare_retries + 1})...')
                        
                        # Reset and retry
                        self.tare_samples = []
                        self.tare_start_time = self.get_clock().now()
                        
                        # Notify manager of retry
                        retry_msg = String()
                        retry_msg.data = f"TARE_RETRY_{self.tare_retry_count}"
                        self.result_pub.publish(retry_msg)
                        return
                    else:
                        # Max retries exceeded
                        self.get_logger().error(f'  ✗ Tare failed after {self.max_tare_retries + 1} attempts')
                        self.get_logger().error('  Is /weight/raw topic publishing?')
                        self.get_logger().error('  Run: ros2 topic echo /weight/raw')
                        
                        # Reset state
                        self.measurement_stage = 'ready'
                        self.tare_samples = []
                        self.tare_retry_count = 0
                        
                        # Notify manager of failure
                        error_msg = String()
                        error_msg.data = "TARE_FAILED"
                        self.result_pub.publish(error_msg)
                        return
            
            # Collecting tare samples (RAW ADC)
            if hasattr(self, 'latest_weight_raw'):
                self.tare_samples.append(self.latest_weight_raw)
                # Log progress every 5 samples
                if len(self.tare_samples) % 5 == 0:
                    self.get_logger().info(f'  Tare progress: {len(self.tare_samples)}/{self.tare_sample_count}')
            else:
                # Log if we don't have weight data yet
                if len(self.tare_samples) == 0:
                    self.get_logger().warn('  Waiting for /weight/raw data...')
            
            if len(self.tare_samples) >= self.tare_sample_count:
                # Tare collection complete
                self.finish_tare()
            
            # If tare didn't collect enough samples within timeout, it will error in finish_tare
            # Do NOT proceed to Stage 2 if tare failed
        
        elif self.sampling_active:
            # Collecting weight samples (RAW ADC)
            self.sampler.add_sample(self.latest_weight_raw)
            progress = self.sampler.get_progress()
            
            # Publish progress
            progress_msg = Float32()
            progress_msg.data = progress
            self.progress_pub.publish(progress_msg)
            
            if self.sampler.is_complete():
                self.finish_sampling()
    

    def finish_sampling(self):
        """Process samples and publish results"""
        current_id = self.measurement_id
        self.get_logger().info(f'■ Sampling complete - Measurement #{current_id}')
        self.sampling_active = False
        
        # STEP 1: Physical plausibility filter (remove impossible weights BEFORE MAD)
        # System specs: 4x 5kg load cells = 20kg max, but with platform mass and safety: 15kg practical limit
        # This catches load cell glitches that produce positive RAW ADC (negative weights)
        all_samples_raw = self.sampler.samples.copy()
        physically_valid_samples = []
        physics_outliers = []
        overweight_samples = []  # Track samples that exceed capacity
        
        for raw_sample in all_samples_raw:
            converted_weight = self.raw_to_weight(raw_sample)
            # Reject physically impossible weights:
            # - Below -100g (platform can't have negative weight - these are load cell glitches)
            # - Above 15000g (exceeds system capacity: 4x 5kg load cells with safety margin)
            if converted_weight < -100.0:
                physics_outliers.append(raw_sample)  # Negative weight glitch
            elif converted_weight > 15000.0:
                overweight_samples.append(raw_sample)  # Overweight
                physics_outliers.append(raw_sample)
            else:
                physically_valid_samples.append(raw_sample)
        
        # Check for genuine overweight condition (multiple consecutive samples, not just glitches)
        if len(overweight_samples) >= 5:  # 5+ samples = 0.5 seconds @ 10 Hz
            self.get_logger().warn('=' * 70)
            self.get_logger().warn('⚠️  OVERWEIGHT WARNING')
            self.get_logger().warn('=' * 70)
            self.get_logger().warn(f'{len(overweight_samples)} samples exceeded 15kg capacity')
            avg_overweight = sum(self.raw_to_weight(s) for s in overweight_samples) / len(overweight_samples)
            self.get_logger().warn(f'Average overweight reading: {avg_overweight:.2f}g')
            self.get_logger().warn('This exceeds the safe capacity of the load cell system.')
            self.get_logger().warn('Remove weight immediately to prevent damage!')
            self.get_logger().warn('=' * 70)
        
        if len(physically_valid_samples) < 5:
            self.get_logger().error('Not enough physically valid samples')
            self.get_logger().error(f'Only {len(physically_valid_samples)}/70 samples were physically plausible')
            if len(overweight_samples) >= 5:
                self.get_logger().error('Most samples indicate OVERWEIGHT condition!')
            return
        
        # Replace samples with physically valid ones for MAD filtering
        self.sampler.samples = physically_valid_samples
        
        # STEP 2: MAD filtering on physically valid samples
        # Get filtered RAW average
        raw_avg, raw_std, mad_outliers = self.sampler.get_filtered_weight()
        
        if raw_avg is None:
            self.get_logger().error('Not enough samples after filtering')
            return
        
        # Total outliers = physics outliers + MAD outliers
        total_outliers = physics_outliers + mad_outliers
        
        # Convert filtered RAW to weight (more stable than converting each sample)
        weight_gross = self.raw_to_weight(raw_avg)
        
        # Subtract tare (both in RAW ADC units)
        raw_tare = self.tare_reading if self.tare_reading else 0.0
        weight_tare = self.raw_to_weight(raw_tare) if raw_tare != 0 else 0.0
        weight_net = weight_gross - weight_tare
        
        # Calculate std dev from RAW std
        # For small variations, weight_std ≈ raw_std / scale_factor
        # Get scale factor from calibration
        try:
            import json
            from pathlib import Path
            cal_file = Path.home() / '.ros' / 'hx711_calibration.json'
            
            if cal_file.exists():
                with open(cal_file, 'r') as f:
                    cal_data = json.load(f)
                
                calibrations = cal_data.get('calibrations', [])
                if calibrations:
                    cal = None
                    active_id = cal_data.get('active_calibration')
                    if active_id:
                        for c in calibrations:
                            if c.get('id') == active_id:
                                cal = c
                                break
                    if cal is None:
                        cal = calibrations[0]
                    
                    # Get average scale factor
                    points = cal.get('points', [])
                    if points:
                        avg_scale = sum(p['scale'] for p in points) / len(points)
                        std = abs(raw_std / avg_scale)
                    else:
                        std = 0.0
                else:
                    std = 0.0
            else:
                std = 0.0
        except:
            std = 0.0
        
        # ALWAYS generate debug data (manager will decide whether to display it)
        debug_lines = []
        debug_lines.append(f'DEBUG: MEASUREMENT #{current_id}')
        debug_lines.append(f'(Two-stage filtering: 1) Physics check, 2) MAD outlier removal)')
        debug_lines.append(f'Tare (RAW): {raw_tare:.0f} ADC' if raw_tare != 0 else 'Tare: None')
        debug_lines.append('')
        
        # Show ORIGINAL samples (before any filtering)
        debug_lines.append(f'Total RAW samples collected: {len(all_samples_raw)}')
        debug_lines.append('')
        debug_lines.append('ALL RAW ADC samples (with converted weight):')
        for i, s in enumerate(all_samples_raw):
            converted = self.raw_to_weight(s)
            is_physics_outlier = s in physics_outliers
            marker = '[PHYSICS OUTLIER]' if is_physics_outlier else ''
            debug_lines.append(f'  Sample {i+1:2d}: RAW={s:.0f} → {converted:8.2f}g {marker}')
        
        # Show statistics on ORIGINAL samples
        import statistics as stats
        debug_lines.append('')
        debug_lines.append('RAW ADC statistics (all samples):')
        debug_lines.append(f'  Mean: {stats.mean(all_samples_raw):.0f} ADC')
        debug_lines.append(f'  Median: {stats.median(all_samples_raw):.0f} ADC')
        debug_lines.append(f'  Std Dev: {stats.stdev(all_samples_raw) if len(all_samples_raw) > 1 else 0:.0f} ADC')
        debug_lines.append(f'  Min: {min(all_samples_raw):.0f} ADC')
        debug_lines.append(f'  Max: {max(all_samples_raw):.0f} ADC')
        debug_lines.append(f'  Range: {max(all_samples_raw) - min(all_samples_raw):.0f} ADC')
        
        # Show Physics filtering results
        debug_lines.append('')
        debug_lines.append('STEP 1: Physical Plausibility Filter')
        debug_lines.append(f'  Valid range: -100g to 15000g (4x 5kg load cells with safety margin)')
        debug_lines.append(f'  Physics outliers removed: {len(physics_outliers)}')
        if overweight_samples:
            debug_lines.append(f'    - Overweight (>15kg): {len(overweight_samples)} samples')
            debug_lines.append(f'    - Negative glitches (<-100g): {len(physics_outliers) - len(overweight_samples)} samples')
        else:
            debug_lines.append(f'    - Negative glitches (<-100g): {len(physics_outliers)} samples')
        debug_lines.append(f'  Physically valid samples: {len(physically_valid_samples)}/{len(all_samples_raw)}')
        
        if physics_outliers:
            debug_lines.append('')
            # Separate negative glitches from overweight
            negative_glitches = [s for s in physics_outliers if self.raw_to_weight(s) < -100.0]
            if negative_glitches:
                debug_lines.append(f'Negative glitches (load cell errors - {len(negative_glitches)} total):')
                for i, s in enumerate(negative_glitches[:10]):  # Show first 10
                    converted = self.raw_to_weight(s)
                    debug_lines.append(f'  ✗ Glitch {i+1:2d}: RAW={s:.0f} → {converted:.2f}g')
                if len(negative_glitches) > 10:
                    debug_lines.append(f'  ... and {len(negative_glitches) - 10} more glitches')
            
            if overweight_samples:
                debug_lines.append('')
                debug_lines.append(f'⚠️  OVERWEIGHT samples (>15kg capacity - {len(overweight_samples)} total):')
                for i, s in enumerate(overweight_samples[:10]):  # Show first 10
                    converted = self.raw_to_weight(s)
                    debug_lines.append(f'  ⚠️  Overweight {i+1:2d}: RAW={s:.0f} → {converted:.2f}g')
                if len(overweight_samples) > 10:
                    debug_lines.append(f'  ... and {len(overweight_samples) - 10} more overweight samples')
        
        # Show MAD filtering details on physically valid samples
        median = stats.median(physically_valid_samples)
        mad = stats.median([abs(s - median) for s in physically_valid_samples])
        threshold = OUTLIER_STD_THRESHOLD * mad
        
        debug_lines.append('')
        debug_lines.append('STEP 2: MAD Outlier Detection (on physically valid samples)')
        debug_lines.append(f'  Median: {median:.0f} ADC')
        debug_lines.append(f'  MAD: {mad:.0f} ADC')
        debug_lines.append(f'  Threshold: ±{threshold:.0f} ADC')
        debug_lines.append(f'  MAD outliers removed: {len(mad_outliers)}')
        
        if mad_outliers:
            debug_lines.append('')
            debug_lines.append(f'MAD outliers (statistical outliers - {len(mad_outliers)} total):')
            for i, s in enumerate(mad_outliers):
                deviation = abs(s - median)
                converted = self.raw_to_weight(s)
                debug_lines.append(f'  ✗ MAD outlier {i+1:2d}: RAW={s:.0f} ({converted:.2f}g) deviation={deviation:.0f} ADC')
        
        # Calculate valid samples after both filters
        valid_samples_final = [s for s in physically_valid_samples if s not in mad_outliers]
        
        if valid_samples_final:
            debug_lines.append('')
            debug_lines.append(f'Final valid samples (after both filters - {len(valid_samples_final)} total):')
            for i, s in enumerate(valid_samples_final[:10]):  # Show first 10
                converted = self.raw_to_weight(s)
                debug_lines.append(f'  ✓ Valid {i+1:2d}: RAW={s:.0f} → {converted:.2f}g')
            if len(valid_samples_final) > 10:
                debug_lines.append(f'  ... and {len(valid_samples_final) - 10} more valid samples')
        
        # Show final filtered result
        debug_lines.append('')
        debug_lines.append('Final filtered result:')
        debug_lines.append(f'  RAW average: {raw_avg:.0f} ADC (filtered)')
        debug_lines.append(f'  Gross weight: {weight_gross:.2f}g (converted from RAW)')
        debug_lines.append(f'  Tare weight: {weight_tare:.2f}g' if weight_tare != 0 else f'  Tare weight: 0.00g (no tare)')
        debug_lines.append(f'  Net weight: {weight_net:.2f}g (gross - tare)')
        debug_lines.append(f'  Std Dev: {std:.2f}g')
        debug_lines.append(f'  Total valid samples: {len(valid_samples_final)}/{len(all_samples_raw)}')
        
        # Add summary section at the end (easy to see without scrolling)
        debug_lines.append('')
        debug_lines.append('=' * 80)
        debug_lines.append('SUMMARY:')
        debug_lines.append('=' * 80)
        if raw_tare != 0:
            debug_lines.append(f'Tare (Stage 1):  {raw_tare:.0f} ADC  →  {weight_tare:.2f}g')
        else:
            debug_lines.append(f'Tare (Stage 1):  Not taken (using 0g)')
        debug_lines.append(f'Gross weight:    {raw_avg:.0f} ADC  →  {weight_gross:.2f}g')
        debug_lines.append(f'Net weight:      {weight_net:.2f}g (±{std:.2f}g)')
        if overweight_samples:
            debug_lines.append(f'')
            debug_lines.append(f'⚠️  OVERWEIGHT:   {len(overweight_samples)} samples exceeded 15kg capacity!')
        debug_lines.append(f'Physics outliers: {len(physics_outliers)} ({len([s for s in physics_outliers if self.raw_to_weight(s) < -100])} glitches, {len(overweight_samples)} overweight)')
        debug_lines.append(f'MAD outliers:     {len(mad_outliers)} (statistical outliers)')
        debug_lines.append(f'Total outliers:   {len(total_outliers)}/{len(all_samples_raw)} samples removed')
        debug_lines.append('=' * 80)
        
        # Publish ALL debug data as a single message to prevent message loss
        # Sending 100+ individual messages can cause ROS2 queue overflow
        debug_msg = String()
        debug_msg.data = '\n'.join(debug_lines)
        self.debug_pub.publish(debug_msg)
        
        # Small delay to ensure debug message is sent before result
        import time
        time.sleep(0.05)
        
        # Get detected items
        items = []
        for (box, score, class_id) in self.latest_detections:
            x, y, bw, bh = box
            cx = x + bw // 2
            cy = y + bh // 2
            distance = self.get_depth_at(cx, cy)
            
            if distance > 0.0:
                label = CLASSES[class_id] if class_id < len(CLASSES) else str(class_id)
                items.append(f'{label}({distance:.2f}m)')
        
        # Verify weight against database if barcode is available
        verification_status = ''
        if self.latest_barcode != 'N/A':
            status, expected, tolerance, message = self.verify_weight(self.latest_barcode, weight_net)
            
            if status == 'MATCH':
                verification_status = f'  ✓ MATCH: {message} (expected: {expected:.0f}g ±{tolerance:.0f}g)'
            elif status == 'MISMATCH':
                verification_status = f'  ✗ MISMATCH: {message} (expected: {expected:.0f}g ±{tolerance:.0f}g)'
            elif status == 'NO_DATA':
                verification_status = f'  ? {message}'
        
        # Build result string
        result_lines = [
            f"  Items detected : {', '.join(items) if items else 'None'}",
            f"  Weight         : {weight_net:.2f}g (±{std:.2f}g)",
            f"  Barcode        : {self.latest_barcode}",
            f"  Outliers       : {len(total_outliers)} samples removed ({len([s for s in physics_outliers if self.raw_to_weight(s) < -100])} glitches, {len(overweight_samples)} overweight, {len(mad_outliers)} MAD)"
        ]
        
        if len(overweight_samples) >= 5:
            result_lines.append(f"  ⚠️  WARNING     : OVERWEIGHT! {len(overweight_samples)} samples exceeded 15kg capacity")
        
        if verification_status:
            result_lines.append(verification_status)
        
        result_str = '\n'.join(result_lines)
        
        # Publish result
        result_msg = String()
        result_msg.data = result_str
        self.result_pub.publish(result_msg)
        
        # Small delay to ensure result message is fully transmitted before manager shows prompt
        import time
        time.sleep(0.1)
        
        self.get_logger().info('Sampling complete')
        
        # Reset to stage 1 for next measurement
        self.measurement_stage = 'ready'
        self.tare_reading = None
        self.get_logger().info('Ready for next measurement (Stage 1: Tare)')

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'Depth error: {e}')

    def barcode_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            return

        barcodes = decode(frame, symbols=[ZBarSymbol.CODE39, ZBarSymbol.CODE128])
        
        # Clear barcode if none detected
        if not barcodes:
            if self.latest_barcode != 'N/A':
                self.latest_barcode = 'N/A'
                # Publish the cleared barcode
                barcode_msg = String()
                barcode_msg.data = 'N/A'
                self.barcode_pub.publish(barcode_msg)
        
        for barcode in barcodes:
            data  = barcode.data.decode('utf-8')
            btype = barcode.type

            if data != self.latest_barcode:
                self.latest_barcode = data
                self.get_logger().info(f'Barcode: {btype}:{data}')
                # Publish new barcode
                barcode_msg = String()
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
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding='passthrough')
            debug_msg.header = msg.header
            self.barcode_debug_pub.publish(debug_msg)
        except:
            pass

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        outputs = np.array(outputs[0]).T

        boxes, scores, class_ids = [], [], []
        for row in outputs:
            classes_score = row[4:]
            max_score = np.max(classes_score)
            if max_score >= CONFIDENCE_THRESHOLD:
                class_id = np.argmax(classes_score)
                cx, cy, bw, bh = row[:4]
                x = int((cx - bw/2) * w / INPUT_SIZE)
                y = int((cy - bh/2) * h / INPUT_SIZE)
                width = int(bw * w / INPUT_SIZE)
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
            frame = self.frame_queue.get()
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
        except:
            return

        if self.frame_count % DETECT_EVERY_N_FRAMES == 0:
            if not self.frame_queue.full():
                try:
                    self.frame_queue.put(color_frame.copy())
                except Exception as e:
                    self.get_logger().warn(f'Frame queue error: {e}')

        if not self.result_queue.empty():
            self.latest_detections = self.result_queue.get()

        # Draw detections
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

        # Publish current items to manager
        items_msg = String()
        items_msg.data = ', '.join(detected_items) if detected_items else 'None'
        self.items_pub.publish(items_msg)

        # Overlay info
        status_text = "SAMPLING" if self.sampling_active else "READY"
        status_color = (0, 165, 255) if self.sampling_active else (0, 255, 0)
        
        # Convert RAW to weight for display
        display_weight = self.raw_to_weight(self.latest_weight_raw)
        
        cv2.putText(color_frame, f'Weight: {display_weight:.1f}g', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(color_frame, f'Barcode: {self.latest_barcode}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
        cv2.putText(color_frame, f'Status: {status_text}', (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        try:
            result_msg = self.bridge.cv2_to_imgmsg(color_frame, encoding='passthrough')
            self.detection_pub.publish(result_msg)
        except:
            pass


def main():
    rclpy.init()
    backend = BinWeighingBackend()
    rclpy.spin(backend)
    backend.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
