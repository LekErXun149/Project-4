"""
HX711 Hardware Driver
Low-level interface to HX711 ADC
"""

import RPi.GPIO as GPIO
import time
import statistics
from typing import List, Tuple, Optional


class HX711Driver:
    """Hardware driver for HX711 24-bit ADC"""
    
    def __init__(self, dout_pin: int = 27, sck_pin: int = 17, gain: int = 128):
        """
        Initialize HX711
        
        Args:
            dout_pin: GPIO pin for data (default: 27)
            sck_pin: GPIO pin for clock (default: 17)
            gain: Amplifier gain - 128 (Ch A), 64 (Ch A), 32 (Ch B)
        """
        self.DOUT = dout_pin
        self.SCK = sck_pin
        self.GAIN = 1 if gain == 128 else (3 if gain == 32 else 2)
        
        # Calibration parameters
        self.offset = 0.0
        self.scale = 1.0
        
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.SCK, GPIO.OUT)
        GPIO.setup(self.DOUT, GPIO.IN)
        GPIO.output(self.SCK, False)
        
        # Power cycle
        self._power_cycle()
        
        # Flush initial readings
        for _ in range(5):
            self.read_raw()
            time.sleep(0.1)
    
    def _power_cycle(self):
        """Power cycle the HX711"""
        GPIO.output(self.SCK, True)
        time.sleep(0.1)
        GPIO.output(self.SCK, False)
        time.sleep(1)
    
    def is_ready(self) -> bool:
        """Check if HX711 has data ready"""
        return GPIO.input(self.DOUT) == 0
    
    def read_raw(self, timeout: float = 1.0) -> Optional[int]:
        """
        Read raw 24-bit value from HX711
        
        Args:
            timeout: Maximum time to wait for ready (seconds)
            
        Returns:
            Raw signed 24-bit value, or None if timeout
        """
        # Wait for ready
        start_time = time.time()
        while not self.is_ready():
            if time.time() - start_time > timeout:
                return None
            time.sleep(0.001)
        
        # Read 24 bits (no delays for RPi timing)
        value = 0
        for _ in range(24):
            GPIO.output(self.SCK, True)
            value = (value << 1) | GPIO.input(self.DOUT)
            GPIO.output(self.SCK, False)
        
        # Send gain pulses
        for _ in range(self.GAIN):
            GPIO.output(self.SCK, True)
            GPIO.output(self.SCK, False)
        
        # Convert to signed 24-bit
        if value & 0x800000:
            value -= 0x1000000
        
        return value
    
    def read_average(self, samples: int = 10, 
                     remove_outliers: bool = True) -> Tuple[float, List[int]]:
        """
        Read average of multiple samples with optional outlier rejection
        
        Args:
            samples: Number of samples to take
            remove_outliers: Whether to remove outliers using MAD
            
        Returns:
            (average_value, list_of_samples)
        """
        values = []
        
        for _ in range(samples):
            val = self.read_raw()
            if val is not None:
                values.append(val)
            time.sleep(0.05)
        
        if len(values) < 3:
            return 0.0, values
        
        if remove_outliers:
            # Remove outliers using Median Absolute Deviation
            median = statistics.median(values)
            mad = statistics.median([abs(v - median) for v in values])
            
            if mad > 0:
                threshold = 2 * mad
                filtered = [v for v in values if abs(v - median) < threshold]
                
                # Keep at least half the samples
                if len(filtered) >= len(values) // 2:
                    values = filtered
        
        return statistics.mean(values), values
    
    def tare(self, samples: int = 20) -> bool:
        """
        Tare (zero) the scale
        
        Args:
            samples: Number of samples for taring
            
        Returns:
            True if successful
        """
        avg, _ = self.read_average(samples, remove_outliers=True)
        
        if avg == 0:
            return False
        
        self.offset = avg
        return True
    
    def calibrate(self, known_weight: float, samples: int = 20) -> bool:
        """
        Calibrate scale with known weight
        
        Args:
            known_weight: Known weight in grams
            samples: Number of samples for calibration
            
        Returns:
            True if successful
        """
        avg, _ = self.read_average(samples, remove_outliers=True)
        reading = avg - self.offset
        
        if abs(reading) < 100:
            return False
        
        self.scale = reading / known_weight
        return True
    
    def get_weight(self, samples: int = 5) -> float:
        """
        Get current weight in grams
        
        Args:
            samples: Number of samples to average
            
        Returns:
            Weight in grams
        """
        avg, _ = self.read_average(samples, remove_outliers=False)
        
        if self.scale == 0:
            return 0.0
        
        return (avg - self.offset) / self.scale
    
    def get_raw_value(self) -> int:
        """Get single raw reading"""
        val = self.read_raw()
        return val if val is not None else 0
    
    def set_calibration(self, offset: float, scale: float):
        """
        Set calibration parameters directly
        
        Args:
            offset: Offset value (from tare)
            scale: Scale factor (units per gram)
        """
        self.offset = offset
        self.scale = scale
    
    def get_calibration(self) -> Tuple[float, float]:
        """
        Get current calibration parameters
        
        Returns:
            (offset, scale)
        """
        return self.offset, self.scale
    
    def cleanup(self):
        """Clean up GPIO"""
        GPIO.cleanup()
