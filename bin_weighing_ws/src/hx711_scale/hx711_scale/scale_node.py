#!/usr/bin/env python3
"""
HX711 Scale Node with Multi-Point Calibration Support
Publishes weight readings using linear interpolation for improved accuracy
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger, SetBool

import time
from .hx711_driver import HX711Driver
from .calibration_manager import CalibrationManager
from .interpolation_logic import calculate_weight_multipoint


class ScaleNode(Node):
    """ROS2 node for HX711 scale with multi-point calibration"""
    
    def __init__(self):
        super().__init__('hx711_scale')
        
        # Parameters
        self.declare_parameter('dout_pin', 27)
        self.declare_parameter('sck_pin', 17)
        self.declare_parameter('gain', 128)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('calibration_file', '')
        
        # Get parameters
        dout = self.get_parameter('dout_pin').value
        sck = self.get_parameter('sck_pin').value
        gain = self.get_parameter('gain').value
        self.rate_hz = self.get_parameter('rate_hz').value
        cal_file = self.get_parameter('calibration_file').value
        
        # Initialize driver
        self.get_logger().info(f'Initializing HX711 (DT={dout}, SCK={sck}, Gain={gain})')
        self.driver = HX711Driver(dout, sck, gain)
        
        # Load calibration
        if cal_file:
            self.cal_mgr = CalibrationManager(cal_file)
        else:
            self.cal_mgr = CalibrationManager()
        
        self.calibration_loaded = self.load_calibration()
        
        # Publishers
        self.weight_pub = self.create_publisher(Float64, '/weight', 10)
        self.weight_raw_pub = self.create_publisher(Float64, '/weight/raw', 10)
        self.wrench_pub = self.create_publisher(WrenchStamped, '/wrench', 10)
        
        # Services
        self.tare_srv = self.create_service(Trigger, '/tare', self.tare_callback)
        self.enable_srv = self.create_service(SetBool, '/enable', self.enable_callback)
        self.reload_srv = self.create_service(Trigger, '/reload_calibration', self.reload_callback)
        
        # State
        self.enabled = True
        
        # Timer
        self.timer = self.create_timer(1.0 / self.rate_hz, self.timer_callback)
        
        self.get_logger().info('=' * 70)
        self.get_logger().info('HX711 Scale Node initialized successfully')
        self.get_logger().info('=' * 70)
        self.get_logger().info(f'Publishing at {self.rate_hz} Hz')
        self.get_logger().info('Topics:')
        self.get_logger().info('  - /weight (Float64) - Weight in grams')
        self.get_logger().info('  - /weight/raw (Float6464) - Raw ADC value')
        self.get_logger().info('  - /wrench (WrenchStamped) - Force in Newtons')
        self.get_logger().info('Services:')
        self.get_logger().info('  - /tare (Trigger) - Zero the scale')
        self.get_logger().info('  - /enable (SetBool) - Enable/disable publishing')
        self.get_logger().info('  - /reload_calibration (Trigger) - Reload from file')
        self.get_logger().info('=' * 70)
    
    def load_calibration(self) -> bool:
        """Load active calibration from file"""
        cal_data = self.cal_mgr.get_active_calibration()
        
        if cal_data is None:
            self.get_logger().error('=' * 70)
            self.get_logger().error('⚠️  NO ACTIVE CALIBRATION FOUND')
            self.get_logger().error('=' * 70)
            self.get_logger().error('Using DEFAULT TEST CALIBRATION as fallback')
            self.get_logger().error('This is for TESTING ONLY - measurements may be INACCURATE!')
            self.get_logger().error('')
            self.get_logger().error('ACTION REQUIRED:')
            self.get_logger().error('  1. Stop the system')
            self.get_logger().error('  2. Run calibration wizard:')
            self.get_logger().error('     ros2 run hx711_scale calibration_node')
            self.get_logger().error('  3. Create a proper calibration')
            self.get_logger().error('=' * 70)
            
            # Load default test calibration from interpolation_logic
            from .interpolation_logic import get_default_calibration
            cal_data = get_default_calibration()
        
        # Load calibration data
        self.offset = cal_data['offset']
        self.calibration_type = cal_data.get('type', 'single_point')
        self.calibration_points = cal_data['points']
        
        # Set driver calibration (for legacy compatibility)
        if self.calibration_type == 'single_point':
            self.scale = self.calibration_points[0]['scale']
            self.driver.set_calibration(self.offset, self.scale)
        else:
            # For multi-point, set average scale (not actually used for weight calc)
            avg_scale = sum(p['scale'] for p in self.calibration_points) / len(self.calibration_points)
            self.scale = avg_scale
            self.driver.set_calibration(self.offset, avg_scale)
        
        # Only log success if we actually loaded from file
        if self.cal_mgr.get_active_calibration() is not None:
            self.get_logger().info(f'✅ Loaded saved calibration from: {self.cal_mgr.get_file_path()}')
            
            if self.calibration_type == 'multi_point':
                self.get_logger().info(f'   Type: Multi-point ({len(self.calibration_points)} points)')
                weights = [p['weight'] for p in self.calibration_points]
                self.get_logger().info(f'   Range: {min(weights):.0f}g - {max(weights):.0f}g')
                self.get_logger().info(f'   Offset: {self.offset:.0f}')
                for p in self.calibration_points:
                    self.get_logger().info(f'     {p["weight"]:.0f}g: scale = {p["scale"]:.2f}')
            else:
                self.get_logger().info(f'   Type: Single-point')
                self.get_logger().info(f'   Offset: {self.offset:.0f}, Scale: {self.scale:.2f}')
            
            return True
        else:
            # Using fallback - already warned above
            return False
    
    def get_weight(self, samples=5) -> float:
        """
        Get weight reading with multi-point interpolation support
        
        Args:
            samples: Number of samples to average
        
        Returns:
            Weight in grams
        """
        readings = []
        for _ in range(samples):
            raw = self.driver.read_raw()
            if raw is not None:
                readings.append(raw)
            time.sleep(0.01)
        
        if not readings:
            return 0.0
        
        raw_avg = sum(readings) / len(readings)
        
        # Use multi-point calibration if available
        if self.calibration_type == 'multi_point':
            weight = calculate_weight_multipoint(
                raw_avg,
                self.offset,
                self.calibration_points,
                iterations=3
            )
        else:
            # Single-point (legacy)
            weight = (raw_avg - self.offset) / self.scale
        
        return weight
    
    def timer_callback(self):
        """Publish weight readings"""
        if not self.enabled:
            return
        
        # Get weight
        weight = self.get_weight(samples=5)
        
        # Publish weight
        msg = Float64()
        msg.data = weight
        self.weight_pub.publish(msg)
        
        # Publish raw
        raw = self.driver.read_raw()
        if raw is not None:
            raw_msg = Float64()
            raw_msg.data = float(raw)
            self.weight_raw_pub.publish(raw_msg)
        
        # Publish wrench (force in Newtons)
        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = self.get_clock().now().to_msg()
        wrench_msg.header.frame_id = 'scale'
        wrench_msg.wrench.force.z = weight * 0.00980665  # grams to Newtons
        self.wrench_pub.publish(wrench_msg)
    
    def tare_callback(self, request, response):
        """Tare (zero) the scale"""
        try:
            readings = []
            for _ in range(20):
                raw = self.driver.read_raw()
                if raw is not None:
                    readings.append(raw)
                time.sleep(0.05)
            
            if readings:
                new_offset = sum(readings) / len(readings)
                self.offset = new_offset
                self.driver.offset = new_offset
                
                response.success = True
                response.message = f'Tared successfully. New offset: {new_offset:.0f}'
                self.get_logger().info(response.message)
            else:
                response.success = False
                response.message = 'Failed to get readings for tare'
                self.get_logger().error(response.message)
        
        except Exception as e:
            response.success = False
            response.message = f'Tare error: {e}'
            self.get_logger().error(response.message)
        
        return response
    
    def enable_callback(self, request, response):
        """Enable/disable publishing"""
        self.enabled = request.data
        
        response.success = True
        response.message = f'Publishing {"enabled" if self.enabled else "disabled"}'
        self.get_logger().info(response.message)
        
        return response
    
    def reload_callback(self, request, response):
        """Reload calibration from file"""
        try:
            # Reload calibration manager
            self.cal_mgr = CalibrationManager(self.cal_mgr.get_file_path())
            
            # Load calibration
            if self.load_calibration():
                response.success = True
                response.message = 'Calibration reloaded successfully'
                self.get_logger().info('✅ ' + response.message)
            else:
                response.success = False
                response.message = 'No active calibration found after reload'
                self.get_logger().warn('⚠️  ' + response.message)
        
        except Exception as e:
            response.success = False
            response.message = f'Reload error: {e}'
            self.get_logger().error(response.message)
        
        return response
    
    def cleanup(self):
        """Cleanup GPIO"""
        self.driver.cleanup()


def main(args=None):
    rclpy.init(args=args)
    
    node = ScaleNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
