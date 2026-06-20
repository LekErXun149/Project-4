#!/usr/bin/env python3
"""
Interactive HX711 Calibration Node
Runs when needed to calibrate the scale
Saves results for scale_node to use
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from std_srvs.srv import Trigger

import sys
import time
import statistics
from .calibration_manager import CalibrationManager
from .hx711_driver import HX711Driver


class CalibrationNode(Node):
    """Interactive calibration tool"""
    
    def __init__(self):
        super().__init__('scale_calibration')
        
        # Parameters
        self.declare_parameter('dout_pin', 27)
        self.declare_parameter('sck_pin', 17)
        self.declare_parameter('gain', 128)
        self.declare_parameter('calibration_file', '')
        
        # Get parameters
        dout = self.get_parameter('dout_pin').value
        sck = self.get_parameter('sck_pin').value
        gain = self.get_parameter('gain').value
        cal_file = self.get_parameter('calibration_file').value
        
        # Initialize driver
        self.get_logger().info('Initializing HX711 for calibration...')
        self.driver = HX711Driver(dout, sck, gain)
        
        # Calibration manager
        if cal_file:
            self.cal_mgr = CalibrationManager(cal_file)
        else:
            self.cal_mgr = CalibrationManager()
        
        self.get_logger().info(f'Calibration file: {self.cal_mgr.get_file_path()}')
    
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "="*70)
        print(text)
        print("="*70)
    
    def wait_for_stable(self, duration=3.0):
        """Wait for platform to stabilize"""
        print(f"Waiting {duration:.0f} seconds for stabilization...")
        time.sleep(duration)
    
    def take_readings(self, samples=20, show_progress=True):
        """Take multiple readings with outlier rejection"""
        readings = []
        
        if show_progress:
            print(f"Taking {samples} readings...")
        
        for i in range(samples):
            val = self.driver.read_raw()
            if val is not None:
                readings.append(val)
                if show_progress:
                    print(f"  Sample {i+1:2d}/{samples}: {val:10.0f}")
            time.sleep(0.1)
        
        if len(readings) < 10:
            print("⚠️  Too few valid readings!")
            return None, []
        
        # Remove outliers using MAD
        median = statistics.median(readings)
        mad = statistics.median([abs(r - median) for r in readings])
        
        if mad > 0:
            threshold = 2 * mad
            filtered = []
            outliers_list = []
            
            for i, r in enumerate(readings):
                deviation = abs(r - median)
                is_outlier = deviation >= threshold
                
                if is_outlier:
                    outliers_list.append((i+1, r, deviation))
                else:
                    filtered.append(r)
            
            print(f"\n  📊 Analysis:")
            print(f"     Median: {median:.0f}, MAD: {mad:.0f}, Threshold: ±{threshold:.0f}")
            
            if outliers_list:
                print(f"     Removed {len(outliers_list)} outlier(s):")
                for idx, val, dev in outliers_list:
                    print(f"       ✗ Sample {idx:2d}: {val:10.0f} (deviation: {dev:.0f})")
            else:
                print(f"     ✅ No outliers - all samples stable")
            
            if len(filtered) < 10:
                print(f"\n  ⚠️  Too many outliers removed! Using all samples.")
                avg = statistics.mean(readings)
                std = statistics.stdev(readings) if len(readings) > 1 else 0
                print(f"     Final: {avg:.0f} (±{std:.0f}) from {len(readings)} samples")
                return avg, readings
            
            avg = statistics.mean(filtered)
            std = statistics.stdev(filtered) if len(filtered) > 1 else 0
            print(f"     Final: {avg:.0f} (±{std:.0f}) from {len(filtered)}/{len(readings)} samples")
            
            return avg, filtered
        
        avg = statistics.mean(readings)
        print(f"\n  Final: {avg:.0f} from {len(readings)} samples (no outlier filtering)")
        return avg, readings
    
    def run_calibration_wizard(self):
        """Run interactive calibration wizard"""
        self.print_header("HX711 SCALE CALIBRATION WIZARD")
        
        print("\nThis wizard will guide you through calibrating your scale.")
        print("You'll be able to save multiple calibrations with different weights.")
        
        while True:
            self.print_header("MAIN MENU")
            print("\nOptions:")
            print("  1. New single-point calibration")
            print("  2. New multi-point calibration (custom weights)")
            print("  3. List existing calibrations")
            print("  4. Set active calibration")
            print("  5. Delete calibration")
            print("  6. Delete ALL calibrations")
            print("  7. Test current calibration")
            print("  8. Exit")
            
            choice = input("\nEnter choice (1-8): ").strip()
            
            if choice == '1':
                self.new_calibration()
            elif choice == '2':
                self.new_multipoint_calibration()
            elif choice == '3':
                self.list_calibrations()
            elif choice == '4':
                self.set_active()
            elif choice == '5':
                self.delete_calibration()
            elif choice == '6':
                self.delete_all_calibrations()
            elif choice == '7':
                self.test_calibration()
            elif choice == '8':
                print("\nExiting calibration wizard...")
                break
            else:
                print("Invalid choice!")
    
    def new_calibration(self):
        """Create new calibration"""
        self.print_header("NEW SINGLE-POINT CALIBRATION")
        
        # Get calibration name
        name = input("\nCalibration name (or ENTER for default, 'cancel' to abort): ").strip()
        if name.lower() == 'cancel':
            print("\n❌ Calibration cancelled")
            return
        if not name:
            name = None
        
        # Get calibration weight
        while True:
            try:
                weight_str = input("\nCalibration weight in grams (e.g., 500, 1000, 2000) or 'cancel': ").strip()
                if weight_str.lower() == 'cancel':
                    print("\n❌ Calibration cancelled")
                    return
                weight = float(weight_str)
                if weight <= 0:
                    print("Weight must be positive!")
                    continue
                break
            except ValueError:
                print("Invalid weight!")
        
        # Tare
        print("\n" + "-"*70)
        print("STEP 1: TARE (Zero the scale)")
        print("-"*70)
        print("\n⚠️  Remove ALL weight from platform")
        response = input("Press ENTER when ready (or 'cancel' to abort): ").strip().lower()
        
        if response == 'cancel':
            print("\n❌ Calibration cancelled")
            return
        
        self.wait_for_stable(3)
        
        print("\nTaring...")
        offset_avg, offset_readings = self.take_readings(20)
        
        if offset_avg is None:
            print("❌ Tare failed!")
            return
        
        offset_std = statistics.stdev(offset_readings) if len(offset_readings) > 1 else 0
        
        print(f"\n✅ Tare complete!")
        print(f"   Offset: {offset_avg:.0f}")
        print(f"   Std Dev: {offset_std:.0f}")
        
        if offset_std > 1000:
            print("   ⚠️  High variation - platform may be unstable")
            proceed = input("   Continue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                return
        
        # Set offset
        self.driver.offset = offset_avg
        
        # Calibrate
        print("\n" + "-"*70)
        print(f"STEP 2: CALIBRATE with {weight}g")
        print("-"*70)
        print(f"\n📍 Place exactly {weight}g on the CENTER of platform")
        print("   Wait for weight to stabilize (2-3 seconds)")
        response = input("Press ENTER when ready (or 'cancel' to abort): ").strip().lower()
        
        if response == 'cancel':
            print("\n❌ Calibration cancelled")
            return
        
        self.wait_for_stable(3)
        
        print("\nMeasuring...")
        cal_avg, cal_readings = self.take_readings(20)
        
        if cal_avg is None:
            print("❌ Calibration failed!")
            return
        
        # Calculate scale factor
        reading = cal_avg - offset_avg
        
        if abs(reading) < 100:
            print(f"\n❌ Reading too small: {reading:.0f}")
            print("   Check that weight is on platform!")
            return
        
        scale = reading / weight
        cal_std = statistics.stdev(cal_readings) if len(cal_readings) > 1 else 0
        
        print(f"\n✅ Calibration complete!")
        print(f"   Raw reading: {reading:.0f}")
        print(f"   Scale factor: {scale:.2f}")
        print(f"   Std Dev: {cal_std:.0f}")
        print(f"   SNR: {abs(reading/cal_std):.1f}" if cal_std > 0 else "")
        
        # Verify
        print("\n" + "-"*70)
        print("STEP 3: VERIFICATION")
        print("-"*70)
        
        print("\nRemove weight...")
        response = input("Press ENTER when weight removed (or 'cancel' to skip verification): ").strip().lower()
        
        if response == 'cancel':
            print("\n⚠️  Verification skipped")
            # Ask if they still want to save without verification
            save = input("Save calibration without verification? (y/n): ").strip().lower()
            if save == 'y':
                if name is None:
                    name = f"{weight}g calibration (unverified)"
                
                cal_id = self.cal_mgr.add_single_point_calibration(
                    calibration_weight=weight,
                    offset=offset_avg,
                    scale=scale,
                    name=name,
                    set_active=True
                )
                
                print(f"\n✅ Calibration saved (unverified)!")
                print(f"   ID: {cal_id}")
                print(f"   Name: {name}")
                print(f"\n📁 Saved to: {self.cal_mgr.get_file_path()}")
            else:
                print("\n❌ Calibration discarded")
            return
        
        self.wait_for_stable(3)
        
        # Use robust sampling for zero test
        print("\nTesting zero reading...")
        zero_avg, zero_readings = self.take_readings(20, show_progress=True)
        
        # Temporary set calibration for testing
        old_offset = self.driver.offset
        old_scale = self.driver.scale
        self.driver.offset = offset_avg
        self.driver.scale = scale
        
        if zero_avg is not None:
            zero_weight = (zero_avg - offset_avg) / scale
            zero_std = statistics.stdev(zero_readings) if len(zero_readings) > 1 else 0
            zero_weight_std = zero_std / abs(scale) if scale != 0 else 0
            print(f"\n  Zero test: {zero_weight:.1f}g (±{zero_weight_std:.1f}g)")
            if abs(zero_weight) > 10:
                print(f"  ⚠️  Zero reading is {zero_weight:.1f}g - should be close to 0g")
        else:
            print("⚠️  Zero test failed")
        
        print(f"\nPlace {weight}g back on platform...")
        input("Press ENTER when ready...")
        
        self.wait_for_stable(3)
        
        # Use robust sampling with outlier filtering for weight test
        print("\nVerifying calibration weight...")
        weight_avg, weight_readings = self.take_readings(20, show_progress=True)
        
        if weight_avg is None:
            print("❌ Verification failed - unable to get stable readings")
            self.driver.scale = old_scale
            return
        
        # Calculate weight from filtered readings
        print(f"\n  💡 Converting raw readings to weight:")
        print(f"     Formula: (raw - offset) / scale")
        print(f"     Offset: {offset_avg:.0f}")
        print(f"     Scale: {scale:.2f}")
        print(f"\n  Weight conversions:")
        
        test_weights = []
        for i, r in enumerate(weight_readings):
            w = (r - offset_avg) / scale
            test_weights.append(w)
            if i < 5 or i >= len(weight_readings) - 5:  # Show first 5 and last 5
                print(f"     Sample {i+1:2d}: ({r:.0f} - {offset_avg:.0f}) / {scale:.2f} = {w:.1f}g")
            elif i == 5:
                print(f"     ... ({len(weight_readings)-10} more samples) ...")
        
        test_avg = statistics.mean(test_weights)
        test_std = statistics.stdev(test_weights) if len(test_weights) > 1 else 0
        
        error = abs(test_avg - weight)
        error_pct = (error / weight) * 100
        
        print(f"\n📊 Verification Results:")
        print(f"   Expected: {weight:.0f}g")
        print(f"   Measured: {test_avg:.1f}g (±{test_std:.1f}g)")
        print(f"   Error: {error:.1f}g ({error_pct:.2f}%)")
        print(f"   Samples: {len(weight_readings)}/{20} (after outlier removal)")
        print(f"   Weight range: {min(test_weights):.1f}g - {max(test_weights):.1f}g")
        
        if error_pct < 1:
            print("   ✅ Excellent accuracy!")
            quality = "excellent"
        elif error_pct < 3:
            print("   ✅ Good accuracy!")
            quality = "good"
        elif error_pct < 5:
            print("   ⚠️  Acceptable accuracy")
            quality = "acceptable"
        else:
            print("   ⚠️  Poor accuracy - consider recalibrating")
            quality = "poor"
        
        # Restore old scale
        self.driver.offset = old_offset
        self.driver.scale = old_scale
        
        # Save calibration
        print("\n" + "-"*70)
        save = input("Save this calibration? (y/n): ").strip().lower()
        
        if save == 'y':
            # Save with metadata
            if name is None:
                name = f"{weight}g calibration ({quality})"
            
            cal_id = self.cal_mgr.add_single_point_calibration(
                calibration_weight=weight,
                offset=offset_avg,
                scale=scale,
                name=name,
                set_active=True
            )
            
            print(f"\n✅ Calibration saved!")
            print(f"   ID: {cal_id}")
            print(f"   Name: {name}")
            print(f"   Set as active calibration")
            print(f"\n📁 Saved to: {self.cal_mgr.get_file_path()}")
            
            self.get_logger().info(f'Calibration saved: {cal_id}')
        else:
            print("\nCalibration discarded")
    
    def new_multipoint_calibration(self):
        """Create new multi-point calibration with custom weights"""
        self.print_header("NEW MULTI-POINT CALIBRATION")
        
        print("\nThis will guide you through creating a multi-point calibration.")
        print("You'll calibrate at multiple weights to improve accuracy across the full range.")
        print("\nRecommended: Calibrate at 3-4 points spread across your typical weight range")
        print("Example: For 0-2000g range, use 500g, 1000g, 1500g, 2000g\n")
        
        # Get calibration name
        name = input("Calibration name (or ENTER for default, 'cancel' to abort): ").strip()
        if name.lower() == 'cancel':
            print("\n❌ Calibration cancelled")
            return
        if not name:
            name = None
        
        # Let user choose calibration weights
        print("\n" + "-"*70)
        print("CALIBRATION WEIGHTS SETUP")
        print("-"*70)
        print("\nHow many calibration points do you want?")
        print("  Recommended: 3-4 points (good balance of accuracy vs time)")
        print("  Typical: 2-10 points")
        print("  Maximum: 50 points (for extreme accuracy or backup purposes)")
        
        while True:
            try:
                num_points_str = input("\nNumber of points (2-50) or 'cancel': ").strip()
                if num_points_str.lower() == 'cancel':
                    print("\n❌ Calibration cancelled")
                    return
                
                num_points = int(num_points_str)
                
                # Minimum: 2 points required for multi-point calibration
                if num_points < 2:
                    print("⚠️  Need at least 2 points for multi-point calibration")
                    print("    (Use single-point calibration for 1 point)")
                    continue
                
                # Maximum: 50 points (allows for extreme accuracy or backup scenarios)
                # Note: 3-4 points is optimal for most use cases
                # More than 10 points rarely improves accuracy significantly
                # but increases calibration time proportionally
                if num_points > 50:
                    print("⚠️  Maximum 50 points allowed")
                    print("    (More points take longer without much accuracy gain)")
                    continue
                
                # Warn if user selects excessive points
                if num_points > 10:
                    print(f"⚠️  {num_points} points will take ~{num_points * 2} minutes to calibrate")
                    confirm = input("    Continue? (yes/no): ").strip().lower()
                    if confirm not in ['yes', 'y']:
                        continue
                
                break
            except ValueError:
                print("Invalid number!")
        
        # Get each weight
        weights = []
        print(f"\nEnter {num_points} calibration weights (in grams):")
        print("Tip: Spread them across your expected weight range")
        
        for i in range(num_points):
            while True:
                try:
                    weight_str = input(f"  Weight {i+1}/{num_points} (or 'cancel'): ").strip()
                    if weight_str.lower() == 'cancel':
                        print("\n❌ Calibration cancelled")
                        return
                    
                    weight = float(weight_str)
                    if weight <= 0:
                        print("    Weight must be positive!")
                        continue
                    if weight in weights:
                        print("    Weight already entered!")
                        continue
                    
                    weights.append(weight)
                    break
                except ValueError:
                    print("    Invalid weight!")
        
        # Sort weights from smallest to largest
        weights.sort()
        
        print(f"\n✅ Calibration points configured:")
        for w in weights:
            print(f"   {w:.0f}g")
        
        confirm = input("\nProceed with these weights? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("\n❌ Calibration cancelled")
            return
        
        # Tare
        print("\n" + "-"*70)
        print("STEP 1: TARE (Zero the scale)")
        print("-"*70)
        print("\n⚠️  Remove ALL weight from platform")
        response = input("Press ENTER when ready (or 'cancel' to abort): ").strip().lower()
        
        if response == 'cancel':
            print("\n❌ Calibration cancelled")
            return
        
        self.wait_for_stable(3)
        
        print("\nTaring...")
        offset_avg, offset_readings = self.take_readings(20)
        
        if offset_avg is None:
            print("❌ Tare failed!")
            return
        
        offset_std = statistics.stdev(offset_readings) if len(offset_readings) > 1 else 0
        
        print(f"\n✅ Tare complete!")
        print(f"   Offset: {offset_avg:.0f}")
        print(f"   Std Dev: {offset_std:.0f}")
        
        if offset_std > 1000:
            print("   ⚠️  High variation - platform may be unstable")
            proceed = input("   Continue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                return
        
        # Set offset
        self.driver.offset = offset_avg
        
        # Calibrate at each weight
        calibration_points = []
        
        for i, weight in enumerate(weights):
            print("\n" + "-"*70)
            print(f"STEP {i+2}: CALIBRATE with {weight}g")
            print("-"*70)
            print(f"\n📍 Place exactly {weight}g on the CENTER of platform")
            print("   Wait for weight to stabilize (2-3 seconds)")
            response = input("Press ENTER when ready (or 'cancel' to abort): ").strip().lower()
            
            if response == 'cancel':
                print(f"\n❌ Calibration cancelled at {weight}g step")
                # Offer to save partial calibration
                if calibration_points:
                    print(f"\n⚠️  Already calibrated {len(calibration_points)} point(s):")
                    for w, s in calibration_points:
                        print(f"   {w}g: scale = {s:.2f}")
                    save_partial = input("\nSave partial calibration? (y/n): ").strip().lower()
                    if save_partial == 'y':
                        cal_id = self.cal_mgr.add_multi_point_calibration(
                            offset=offset_avg,
                            points=calibration_points,
                            name=f"Partial multi-point ({len(calibration_points)} points, incomplete)",
                            set_active=False
                        )
                        print(f"\n✅ Partial calibration saved as: {cal_id}")
                        print("   ⚠️  NOT set as active (incomplete)")
                return
            
            self.wait_for_stable(3)
            
            print(f"\nMeasuring {weight}g...")
            cal_avg, cal_readings = self.take_readings(20)
            
            if cal_avg is None:
                print(f"❌ Calibration failed at {weight}g!")
                return
            
            # Calculate scale factor for this weight
            reading = cal_avg - offset_avg
            
            if abs(reading) < 100:
                print(f"\n❌ Reading too small: {reading:.0f}")
                print("   Check that weight is on platform!")
                return
            
            scale = reading / weight
            cal_std = statistics.stdev(cal_readings) if len(cal_readings) > 1 else 0
            
            print(f"\n✅ {weight}g calibration complete!")
            print(f"   Raw reading: {reading:.0f}")
            print(f"   Scale factor: {scale:.2f}")
            print(f"   Std Dev: {cal_std:.0f}")
            
            # Save this point
            calibration_points.append((weight, scale))
            
            # Pause before next weight
            if i < len(weights) - 1:
                print(f"\nRemove {weight}g weight...")
                input("Press ENTER to continue...")
        
        # Verification
        print("\n" + "-"*70)
        print(f"STEP {len(weights)+2}: VERIFICATION")
        print("-"*70)
        
        print("\nRemove all weights...")
        response = input("Press ENTER to verify (or 'skip' to skip verification): ").strip().lower()
        
        if response == 'skip':
            print("\n⚠️  Verification skipped")
            # Save without verification
            save = input("Save calibration without verification? (y/n): ").strip().lower()
            if save == 'y':
                if name is None:
                    min_weight = min(weights)
                    max_weight = max(weights)
                    name = f"Multi-point calibration ({min_weight:.0f}-{max_weight:.0f}g, unverified)"
                
                cal_id = self.cal_mgr.add_multi_point_calibration(
                    offset=offset_avg,
                    points=calibration_points,
                    name=name,
                    set_active=True
                )
                
                print(f"\n✅ Multi-point calibration saved (unverified)!")
                print(f"   ID: {cal_id}")
                print(f"   Points: {len(calibration_points)}")
                print(f"\n📁 Saved to: {self.cal_mgr.get_file_path()}")
            else:
                print("\n❌ Calibration discarded")
            return
        
        self.wait_for_stable(3)
        
        # Test at multiple points
        test_results = []
        
        for weight in weights:
            print(f"\nTesting {weight}g...")
            print(f"Place {weight}g on platform...")
            input("Press ENTER when ready...")
            
            self.wait_for_stable(3)
            
            # Take readings
            test_avg, test_readings = self.take_readings(20, show_progress=False)
            
            if test_avg is None:
                print(f"⚠️  Verification failed at {weight}g")
                continue
            
            # Find scale factor for this weight
            scale_for_weight = None
            for w, s in calibration_points:
                if w == weight:
                    scale_for_weight = s
                    break
            
            if scale_for_weight:
                measured_weight = (test_avg - offset_avg) / scale_for_weight
                error = abs(measured_weight - weight)
                error_pct = (error / weight) * 100
                
                print(f"  Expected: {weight:.0f}g")
                print(f"  Measured: {measured_weight:.1f}g")
                print(f"  Error: {error:.1f}g ({error_pct:.2f}%)")
                
                test_results.append(error_pct)
            
            if weight != weights[-1]:
                print("Remove weight...")
                input("Press ENTER to continue...")
        
        # Overall quality
        if test_results:
            avg_error = sum(test_results) / len(test_results)
            max_error = max(test_results)
            
            print(f"\n📊 Overall Verification:")
            print(f"   Average error: {avg_error:.2f}%")
            print(f"   Max error: {max_error:.2f}%")
            
            if max_error < 1:
                quality = "excellent"
                print("   ✅ Excellent accuracy!")
            elif max_error < 3:
                quality = "good"
                print("   ✅ Good accuracy!")
            elif max_error < 5:
                quality = "acceptable"
                print("   ⚠️  Acceptable accuracy")
            else:
                quality = "poor"
                print("   ⚠️  Poor accuracy - consider recalibrating")
        else:
            quality = "unknown"
        
        # Save calibration
        print("\n" + "-"*70)
        save = input("Save this multi-point calibration? (y/n): ").strip().lower()
        
        if save == 'y':
            # Save with metadata
            if name is None:
                min_weight = min(w for w, s in calibration_points)
                max_weight = max(w for w, s in calibration_points)
                name = f"Multi-point calibration ({min_weight:.0f}-{max_weight:.0f}g, {quality})"
            
            cal_id = self.cal_mgr.add_multi_point_calibration(
                offset=offset_avg,
                points=calibration_points,
                name=name,
                set_active=True
            )
            
            print(f"\n✅ Multi-point calibration saved!")
            print(f"   ID: {cal_id}")
            print(f"   Name: {name}")
            print(f"   Points: {len(calibration_points)}")
            weights_used = [p[0] for p in calibration_points]
            print(f"   Weights: {', '.join(f'{w:.0f}g' for w in weights_used)}")
            print(f"   Set as active calibration")
            print(f"\n📁 Saved to: {self.cal_mgr.get_file_path()}")
            
            self.get_logger().info(f'Multi-point calibration saved: {cal_id}')
        else:
            print("\nCalibration discarded")
    
    def list_calibrations(self):
        """List all saved calibrations"""
        self.print_header("SAVED CALIBRATIONS")
        
        cals = self.cal_mgr.list_calibrations()
        active_id = self.cal_mgr.calibration_data.get('active_calibration')
        
        if not cals:
            print("\nNo calibrations saved yet.")
            return
        
        print(f"\n{'ID':<10} {'Name':<40} {'Type':<15} {'Active':<10}")
        print("-"*75)
        
        for cal in cals:
            is_active = "✓" if cal['id'] == active_id else ""
            cal_type = cal.get('type', 'single_point')
            type_display = f"{cal_type} ({len(cal['points'])} pts)" if cal_type == 'multi_point' else 'single_point'
            print(f"{cal['id']:<10} {cal['name']:<40} {type_display:<15} {is_active:<10}")
        
        print(f"\nTotal: {len(cals)} calibration(s)")
        
        # Show details option
        show_detail = input("\nShow details for a calibration? (enter ID or ENTER to skip): ").strip()
        
        if show_detail:
            cal = self.cal_mgr.get_calibration_info(show_detail)
            if cal:
                print("\n" + "-"*70)
                print(f"Calibration: {cal['name']}")
                print("-"*70)
                print(f"ID: {cal['id']}")
                print(f"Created: {cal['timestamp']}")
                print(f"Type: {cal.get('type', 'single_point')}")
                print(f"Offset: {cal['offset']:.0f}")
                
                if cal.get('type') == 'multi_point':
                    print(f"Calibration points:")
                    for p in cal['points']:
                        print(f"  {p['weight']:.0f}g: scale = {p['scale']:.2f}")
                else:
                    print(f"Calibration weight: {cal['points'][0]['weight']}g")
                    print(f"Scale: {cal['points'][0]['scale']:.2f}")
            else:
                print(f"Calibration '{show_detail}' not found")
    
    def set_active(self):
        """Set active calibration"""
        self.print_header("SET ACTIVE CALIBRATION")
        
        cals = self.cal_mgr.list_calibrations()
        
        if not cals:
            print("\nNo calibrations available!")
            return
        
        # Show list
        print(f"\n{'ID':<10} {'Name':<40}")
        print("-"*50)
        for cal in cals:
            print(f"{cal['id']:<10} {cal['name']:<40}")
        
        cal_id = input("\nEnter calibration ID to activate: ").strip()
        
        if self.cal_mgr.set_active_calibration(cal_id):
            print(f"\n✅ Activated: {cal_id}")
            self.get_logger().info(f'Active calibration set to: {cal_id}')
        else:
            print(f"\n❌ Calibration '{cal_id}' not found")
    
    def delete_calibration(self):
        """Delete a calibration"""
        self.print_header("DELETE CALIBRATION")
        
        cals = self.cal_mgr.list_calibrations()
        
        if not cals:
            print("\nNo calibrations to delete!")
            return
        
        # Show list
        print(f"\n{'ID':<10} {'Name':<40}")
        print("-"*50)
        for cal in cals:
            print(f"{cal['id']:<10} {cal['name']:<40}")
        
        cal_id = input("\nEnter calibration ID to delete: ").strip()
        
        confirm = input(f"⚠️  Delete '{cal_id}'? This cannot be undone! (yes/no): ").strip().lower()
        
        if confirm == 'yes':
            if self.cal_mgr.delete_calibration(cal_id):
                print(f"\n✅ Deleted: {cal_id}")
                self.get_logger().info(f'Calibration deleted: {cal_id}')
            else:
                print(f"\n❌ Calibration '{cal_id}' not found")
        else:
            print("\nDeletion cancelled")
    
    def delete_all_calibrations(self):
        """Delete ALL calibrations"""
        self.print_header("DELETE ALL CALIBRATIONS")
        
        cals = self.cal_mgr.list_calibrations()
        
        if not cals:
            print("\nNo calibrations to delete!")
            return
        
        print(f"\n⚠️  WARNING: This will delete ALL {len(cals)} calibration(s)!")
        print("This action CANNOT be undone!\n")
        
        # Show what will be deleted
        print("Calibrations to be deleted:")
        for cal in cals:
            cal_type = cal.get('type', 'single_point')
            print(f"  - {cal['id']}: {cal['name']} ({cal_type})")
        
        print()
        confirm = input(f"⚠️  Type 'DELETE ALL' to confirm (or anything else to cancel): ").strip()
        
        if confirm == 'DELETE ALL':
            # Delete all calibrations
            deleted_count = 0
            for cal in cals[:]:  # Copy list since we're modifying it
                if self.cal_mgr.delete_calibration(cal['id']):
                    deleted_count += 1
            
            print(f"\n✅ Deleted {deleted_count} calibration(s)")
            self.get_logger().info(f'All calibrations deleted: {deleted_count} total')
        else:
            print("\nDeletion cancelled")
    
    def test_calibration(self):
        """Test active calibration"""
        self.print_header("TEST ACTIVE CALIBRATION")
        
        cal_data = self.cal_mgr.get_active_calibration()
        
        if cal_data is None:
            print("\n⚠️  No active calibration!")
            print("   Create and activate a calibration first")
            return
        
        offset = cal_data['offset']
        cal_type = cal_data.get('type', 'single_point')
        
        print(f"\nActive calibration:")
        print(f"  Type: {cal_type}")
        print(f"  Offset: {offset:.0f}")
        
        if cal_type == 'multi_point':
            print(f"  Points: {len(cal_data['points'])}")
            for p in cal_data['points']:
                print(f"    {p['weight']:.0f}g: scale = {p['scale']:.2f}")
        else:
            scale = cal_data['points'][0]['scale']
            print(f"  Scale: {scale:.2f}")
            # Set calibration for single-point
            self.driver.set_calibration(offset, scale)
        
        print("\n📊 Live weight readings (press Ctrl+C to stop):")
        print("\nPlace/remove weights to test...")
        print()
        
        try:
            while True:
                if cal_type == 'multi_point':
                    # Use interpolation for multi-point
                    from .interpolation_logic import calculate_weight_multipoint
                    raw = self.driver.read_raw()
                    if raw is not None:
                        weight = calculate_weight_multipoint(
                            raw,
                            offset,
                            cal_data['points'],
                            iterations=3
                        )
                    else:
                        weight = 0.0
                else:
                    # Use standard get_weight for single-point
                    weight = self.driver.get_weight(5)
                
                bar_len = int(abs(weight) / 100)
                bar = "█" * min(bar_len, 40)
                print(f"\rWeight: {weight:8.1f}g  {bar}     ", end="", flush=True)
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n\nTest stopped")
    
    def cleanup(self):
        """Cleanup"""
        self.driver.cleanup()


def main(args=None):
    rclpy.init(args=args)
    
    node = CalibrationNode()
    
    try:
        node.run_calibration_wizard()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    except Exception as e:
        node.get_logger().error(f'Calibration error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
