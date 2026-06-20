#!/usr/bin/env python3
"""
Bin Weighing Manager (Program 1 - User Interface)
Clean terminal interface for user interaction
Handles:
  - Calibration checking and prompting
  - User trigger for weighing
  - Display results clearly
  - No ROS2 logging spam
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Bool, Empty
from std_srvs.srv import Trigger
import sys
import os
import json
import subprocess
from datetime import datetime

# ── Calibration settings ─────────────────────────────────────────────────────
CALIBRATION_FILE = '/root/.ros/hx711_calibration.json'
CALIBRATION_MAX_AGE_HOURS = 2

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_header():
    """Print application header"""
    clear_screen()
    print(f"{CYAN}{'=' * 80}{RESET}")
    print(f"{BOLD}{CYAN}BIN WEIGHING SYSTEM{RESET}")
    print(f"{CYAN}{'=' * 80}{RESET}\n")


def check_calibration_age():
    """Check if calibration file exists and is recent"""
    if not os.path.exists(CALIBRATION_FILE):
        return False, False, None, None
    
    try:
        with open(CALIBRATION_FILE, 'r') as f:
            cal_data = json.load(f)
        
        file_mtime = os.path.getmtime(CALIBRATION_FILE)
        cal_datetime = datetime.fromtimestamp(file_mtime)
        age = datetime.now() - cal_datetime
        age_hours = age.total_seconds() / 3600
        
        is_recent = age_hours <= CALIBRATION_MAX_AGE_HOURS
        
        return True, is_recent, age_hours, cal_data
        
    except Exception as e:
        print(f"{RED}Error reading calibration: {e}{RESET}")
        return False, False, None, None


def startup_calibration_check():
    """Check calibration and prompt user if needed"""
    
    exists, is_recent, age_hours, cal_data = check_calibration_age()
    
    if not exists:
        print(f"{RED}✗ CALIBRATION MISSING!{RESET}")
        print(f"{YELLOW}  Location: {CALIBRATION_FILE}{RESET}")
        print(f"{RED}  Scale must be calibrated before use.{RESET}")
        print(f"\n{YELLOW}⚠️  WARNING: System will use TEST calibration as fallback{RESET}")
        print(f"{YELLOW}  Measurements will be INACCURATE and FOR TESTING ONLY!{RESET}\n")
        
        response = input(f"{BOLD}{CYAN}>>> Launch calibration wizard? (yes/no): {RESET}").strip().lower()
        
        if response in ['yes', 'y']:
            launch_calibration()
            exists, is_recent, age_hours, cal_data = check_calibration_age()
        else:
            print(f"\n{RED}⚠️  Continuing in TEST MODE - readings are NOT accurate!{RESET}\n")
    
    elif not is_recent:
        print(f"{GREEN}✓ Calibration file found{RESET}")
        print(f"  Age: {age_hours:.1f} hours old")
        print(f"  {YELLOW}⚠  Older than {CALIBRATION_MAX_AGE_HOURS} hours - recalibration recommended{RESET}\n")
        
        response = input(f"{BOLD}{CYAN}>>> Launch calibration wizard? (yes/no): {RESET}").strip().lower()
        
        if response in ['yes', 'y']:
            launch_calibration()
        else:
            print(f"\n{YELLOW}Using existing calibration{RESET}\n")
    
    else:
        print(f"{GREEN}✓ Calibration OK ({age_hours:.1f} hours old){RESET}\n")
    
    input(f"{CYAN}Press ENTER to start weighing system...{RESET}")


def launch_calibration():
    """Launch calibration wizard and reload"""
    print(f"\n{CYAN}Launching calibration wizard...{RESET}\n")
    
    try:
        # Need sudo for GPIO access
        result = subprocess.run([
            'sudo', 'bash', '-c',
            'source /opt/ros/jazzy/setup.bash && '
            'source /home/group5/bin_weighing_ws/install/setup.bash && '
            'ros2 run hx711_scale calibration_node'
        ])
        
        if result.returncode == 0:
            print(f"\n{GREEN}✓ Calibration complete!{RESET}")
            print(f"{CYAN}Reloading calibration in scale node...{RESET}")
            
            # Reload calibration in running scale node
            subprocess.run([
                'ros2', 'service', 'call', '/reload_calibration', 'std_srvs/srv/Trigger'
            ], capture_output=True, timeout=5)
            
            print(f"{GREEN}✓ Calibration reloaded{RESET}\n")
        else:
            print(f"{YELLOW}⚠  Calibration failed (exit code {result.returncode}){RESET}\n")
        
    except Exception as e:
        print(f"{YELLOW}⚠  Calibration error: {e}{RESET}\n")


class BinWeighingManager(Node):
    """User interface manager node"""
    
    def __init__(self):
        super().__init__('bin_weighing_manager')
        
        # Publishers
        self.trigger_pub = self.create_publisher(Empty, '/start_weighing', 10)
        
        # Subscribers
        self.progress_sub = self.create_subscription(
            Float32, '/weighing_progress', self.progress_callback, 10)
        self.result_sub = self.create_subscription(
            String, '/weighing_result', self.result_callback, 10)
        self.items_sub = self.create_subscription(
            String, '/current_items', self.items_callback, 10)
        
        # Debug subscriber (only processes if debug_mode is enabled)
        self.debug_sub = self.create_subscription(
            String, '/debug_sampling', self.debug_callback, 10)
        
        # State
        self.current_items = "None"
        self.sampling_active = False
        self.last_progress = 0
        self.debug_buffer = []  # Store debug messages
        self.debug_mode = False  # Internal debug toggle
        
        self.get_logger().info('Manager started')
    
    def items_callback(self, msg):
        """Update current detected items"""
        self.current_items = msg.data
    
    def debug_callback(self, msg):
        """Collect debug messages (may contain multiple lines)"""
        if self.debug_mode:
            # Split batched messages into individual lines
            lines = msg.data.split('\n')
            self.debug_buffer.extend(lines)
    
    def progress_callback(self, msg):
        """Show sampling progress"""
        progress = int(msg.data)
        
        if progress != self.last_progress:
            self.last_progress = progress
            
            # Progress bar
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\rSampling weight... {bar} {progress}%", end='', flush=True)
            
            if progress >= 100:
                print()  # New line when complete
    
    def result_callback(self, msg):
        """Display weighing results"""
        self.sampling_active = False
        
        # Parse result
        result_data = msg.data
        
        # Handle tare retry
        if result_data.startswith("TARE_RETRY_"):
            retry_num = result_data.split("_")[-1]
            print(f"\n{YELLOW}⟳ Tare timeout - retrying (attempt {int(retry_num) + 1}/3)...{RESET}")
            self.sampling_active = True  # Keep waiting
            return
        
        # Handle tare failure
        if result_data == "TARE_FAILED":
            print(f"\n{RED}✗ Tare Failed after 3 attempts!{RESET}")
            print(f"{RED}Check if scale_node is running and /weight/raw is publishing{RESET}")
            print(f"{YELLOW}Troubleshooting:{RESET}")
            print(f"  1. Check scale_node: ros2 node list | grep scale")
            print(f"  2. Check topic: ros2 topic echo /weight/raw")
            print(f"  3. Restart scale_node if needed\n")
            return
        
        # Special handling for tare completion (Stage 1)
        if result_data == "TARE_COMPLETE":
            # Show brief confirmation that Stage 1 is complete
            print(f"\n{GREEN}✓ Stage 1 Complete: Tare reading taken{RESET}")
            
            # Show debug info for Stage 1 if debug mode is enabled
            if self.debug_mode and self.debug_buffer:
                print(f"\n{YELLOW}{'═' * 80}{RESET}")
                print(f"{YELLOW}{BOLD}🔍 DEBUG: STAGE 1 - TARE DETAILS{RESET}")
                print(f"{YELLOW}{'═' * 80}{RESET}")
                for line in self.debug_buffer:
                    print(line)
                print(f"{YELLOW}{'═' * 80}{RESET}\n")
                
                # Clear debug buffer for next stage
                self.debug_buffer = []
            
            print(f"{CYAN}→ Place item on platform and press ENTER for Stage 2{RESET}")
            print(f"{YELLOW}  Or press 'R' to retake tare if quality seems poor{RESET}\n")
            return
        
        print(f"\n{CYAN}{'─' * 80}{RESET}")
        print(f"{BOLD}RESULTS:{RESET}")
        print(result_data)
        print(f"{CYAN}{'─' * 80}{RESET}\n")
        
        # Show debug info if debug_mode is enabled and we have debug data
        if self.debug_mode and self.debug_buffer:
            print(f"\n{YELLOW}{'═' * 80}{RESET}")
            print(f"{YELLOW}{BOLD}🔍 DEBUG: WEIGHT SAMPLING DETAILS{RESET}")
            print(f"{YELLOW}{'═' * 80}{RESET}")
            for line in self.debug_buffer:
                print(f"{YELLOW}{line}{RESET}")
            print(f"{YELLOW}{'═' * 80}{RESET}\n")
            self.debug_buffer = []  # Clear buffer for next sampling
    
    def trigger_weighing(self):
        """Trigger measurement stage (tare or weigh depending on backend state)"""
        self.sampling_active = True
        self.last_progress = 0
        
        # Clear debug buffer from any previous measurement
        self.debug_buffer = []
        
        msg = Empty()
        self.trigger_pub.publish(msg)


def main():
    # Startup calibration check
    print_header()
    startup_calibration_check()
    
    # Initialize ROS2
    rclpy.init()
    manager = BinWeighingManager()
    
    # Clear screen for clean interface
    print_header()
    print(f"{GREEN}System ready!{RESET}\n")
    
    # User interaction loop
    import threading
    
    def ros_spin():
        """ROS2 spin in background thread"""
        rclpy.spin(manager)
    
    spin_thread = threading.Thread(target=ros_spin, daemon=True)
    spin_thread.start()
    
    # Main user loop
    try:
        print(f"\n{BOLD}{YELLOW}TWO-STAGE MEASUREMENT:{RESET}")
        print(f"  {GREEN}Stage 1:{RESET} Platform EMPTY → Press ENTER (takes tare)")
        print(f"  {GREEN}Stage 2:{RESET} Platform WITH ITEM → Press ENTER (measures weight)\n")
        
        while True:
            # Show current status
            print(f"Current items detected: {manager.current_items}")
            
            # Show debug mode status
            debug_status = f"{GREEN}ON{RESET}" if manager.debug_mode else f"{RED}OFF{RESET}"
            print(f"Debug mode: {debug_status}")
            
            # Wait for user
            user_input = input(f"\n{BOLD}{CYAN}>>> Press ENTER to continue, 'D' for debug toggle, 'R' to retry tare, 'Q' to quit: {RESET}").strip().upper()
            
            if user_input == 'Q':
                print("\nShutting down...")
                break
            elif user_input == 'D':
                manager.debug_mode = not manager.debug_mode
                status = f"{GREEN}ENABLED{RESET}" if manager.debug_mode else f"{RED}DISABLED{RESET}"
                print(f"\n{YELLOW}🔍 Debug mode {status}{RESET}")
                if manager.debug_mode:
                    print(f"{YELLOW}   Will show detailed sampling data on next measurement{RESET}\n")
                continue
            elif user_input == 'R':
                print(f"\n{YELLOW}⟳ Retrying tare (starting from Stage 1)...{RESET}\n")
                # Trigger a new tare
                manager.trigger_weighing()
                # Wait for it to complete
                while manager.sampling_active:
                    import time
                    time.sleep(0.1)
                continue
            
            # Trigger measurement stage (ENTER pressed)
            manager.trigger_weighing()
            
            # Wait for sampling to complete
            while manager.sampling_active:
                import time
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Shutting down...{RESET}")
    finally:
        manager.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
