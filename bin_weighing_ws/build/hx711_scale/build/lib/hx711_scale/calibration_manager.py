"""
Calibration Manager with Multi-Point Support
Handles saving/loading calibration data with multiple calibration points
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime


class CalibrationManager:
    """Manages HX711 calibration data persistence with multi-point support"""
    
    def __init__(self, calibration_file: Optional[str] = None):
        """
        Initialize calibration manager
        
        Args:
            calibration_file: Path to calibration file (JSON or YAML)
                             If None, uses default location
        """
        if calibration_file is None:
            # Default: ~/.ros/hx711_calibration.json
            home = Path.home()
            ros_dir = home / '.ros'
            ros_dir.mkdir(exist_ok=True)
            self.calibration_file = ros_dir / 'hx711_calibration.json'
        else:
            self.calibration_file = Path(calibration_file)
        
        self.calibration_data = self._load_or_create()
    
    def _load_or_create(self) -> Dict:
        """Load existing calibration or create new"""
        if self.calibration_file.exists():
            return self._load()
        else:
            return self._create_default()
    
    def _create_default(self) -> Dict:
        """Create default calibration data"""
        return {
            'version': '2.0',  # Version 2.0 supports multi-point
            'created': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'calibrations': [],
            'active_calibration': None
        }
    
    def _load(self) -> Dict:
        """Load calibration from file"""
        try:
            with open(self.calibration_file, 'r') as f:
                if self.calibration_file.suffix == '.yaml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Migrate old format to new if needed
            if 'version' not in data:
                data['version'] = '1.0'
            
            return data
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return self._create_default()
    
    def _save(self):
        """Save calibration to file"""
        self.calibration_data['last_modified'] = datetime.now().isoformat()
        
        try:
            with open(self.calibration_file, 'w') as f:
                if self.calibration_file.suffix == '.yaml':
                    yaml.dump(self.calibration_data, f, default_flow_style=False)
                else:
                    json.dump(self.calibration_data, f, indent=2)
        except Exception as e:
            print(f"Error saving calibration: {e}")
    
    def add_single_point_calibration(self, 
                                     calibration_weight: float,
                                     offset: float, 
                                     scale: float,
                                     name: Optional[str] = None,
                                     set_active: bool = True) -> str:
        """
        Add single-point calibration (legacy format)
        
        Args:
            calibration_weight: Weight used for calibration (grams)
            offset: Offset value from tare
            scale: Scale factor (raw units per gram)
            name: Optional name for this calibration
            set_active: Set as active calibration
            
        Returns:
            Calibration ID
        """
        # Generate ID
        cal_id = f"cal_{len(self.calibration_data['calibrations'])}"
        
        if name is None:
            name = f"Calibration {calibration_weight}g"
        
        calibration = {
            'id': cal_id,
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'type': 'single_point',
            'offset': offset,
            'points': [
                {
                    'weight': calibration_weight,
                    'scale': scale
                }
            ]
        }
        
        self.calibration_data['calibrations'].append(calibration)
        
        if set_active:
            self.calibration_data['active_calibration'] = cal_id
        
        self._save()
        return cal_id
    
    def add_multi_point_calibration(self,
                                    offset: float,
                                    points: List[Tuple[float, float]],
                                    name: Optional[str] = None,
                                    set_active: bool = True) -> str:
        """
        Add multi-point calibration
        
        Args:
            offset: Offset value from tare (common for all points)
            points: List of (weight, scale) tuples
            name: Optional name for this calibration
            set_active: Set as active calibration
            
        Returns:
            Calibration ID
        """
        cal_id = f"cal_{len(self.calibration_data['calibrations'])}"
        
        if name is None:
            weights = [p[0] for p in points]
            name = f"Multi-point calibration ({min(weights):.0f}-{max(weights):.0f}g)"
        
        calibration = {
            'id': cal_id,
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'type': 'multi_point',
            'offset': offset,
            'points': [
                {
                    'weight': weight,
                    'scale': scale
                }
                for weight, scale in sorted(points)
            ]
        }
        
        self.calibration_data['calibrations'].append(calibration)
        
        if set_active:
            self.calibration_data['active_calibration'] = cal_id
        
        self._save()
        return cal_id
    
    def get_active_calibration(self) -> Optional[Dict]:
        """
        Get active calibration
        
        Returns:
            Calibration dict or None if no active calibration
        """
        active_id = self.calibration_data.get('active_calibration')
        
        if active_id is None:
            return None
        
        for cal in self.calibration_data['calibrations']:
            if cal['id'] == active_id:
                return cal
        
        return None
    
    def set_active_calibration(self, cal_id: str) -> bool:
        """Set active calibration by ID"""
        for cal in self.calibration_data['calibrations']:
            if cal['id'] == cal_id:
                self.calibration_data['active_calibration'] = cal_id
                self._save()
                return True
        return False
    
    def list_calibrations(self) -> list:
        """List all calibrations"""
        return self.calibration_data['calibrations']
    
    def delete_calibration(self, cal_id: str) -> bool:
        """Delete a calibration"""
        for i, cal in enumerate(self.calibration_data['calibrations']):
            if cal['id'] == cal_id:
                self.calibration_data['calibrations'].pop(i)
                
                if self.calibration_data['active_calibration'] == cal_id:
                    self.calibration_data['active_calibration'] = None
                
                self._save()
                return True
        return False
    
    def get_calibration_info(self, cal_id: str) -> Optional[Dict]:
        """Get calibration details"""
        for cal in self.calibration_data['calibrations']:
            if cal['id'] == cal_id:
                return cal
        return None
    
    def get_file_path(self) -> str:
        """Get calibration file path"""
        return str(self.calibration_file)
