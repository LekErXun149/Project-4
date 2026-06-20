"""
Linear Interpolation Logic for Multi-Point Calibration
"""

def interpolate_scale(weight_estimate: float, calibration_points: list) -> float:
    """
    Linear interpolation between two closest calibration points
    
    Args:
        weight_estimate: Estimated weight (grams)
        calibration_points: List of {'weight': float, 'scale': float} dicts
                           Must be sorted by weight
    
    Returns:
        Interpolated scale factor
    """
    # Single point calibration
    if len(calibration_points) == 1:
        return calibration_points[0]['scale']
    
    # Find bracket points
    # If weight is below lowest point, use lowest
    if weight_estimate <= calibration_points[0]['weight']:
        return calibration_points[0]['scale']
    
    # If weight is above highest point, use highest
    if weight_estimate >= calibration_points[-1]['weight']:
        return calibration_points[-1]['scale']
    
    # Find two points that bracket the weight
    for i in range(len(calibration_points) - 1):
        w1 = calibration_points[i]['weight']
        w2 = calibration_points[i + 1]['weight']
        
        if w1 <= weight_estimate <= w2:
            # Linear interpolation
            s1 = calibration_points[i]['scale']
            s2 = calibration_points[i + 1]['scale']
            
            # Interpolation formula: s = s1 + (w - w1) / (w2 - w1) * (s2 - s1)
            ratio = (weight_estimate - w1) / (w2 - w1)
            scale_interpolated = s1 + ratio * (s2 - s1)
            
            return scale_interpolated
    
    # Fallback (shouldn't reach here)
    return calibration_points[0]['scale']


def calculate_weight_multipoint(raw_value: float, offset: float, 
                                calibration_points: list, 
                                iterations: int = 3) -> float:
    """
    Calculate weight using multi-point calibration with iterative refinement
    
    Args:
        raw_value: Raw ADC reading
        offset: Tare offset
        calibration_points: List of {'weight': float, 'scale': float} dicts
        iterations: Number of refinement iterations (default 3)
    
    Returns:
        Calculated weight in grams
    """
    # Initial estimate using average scale
    avg_scale = sum(p['scale'] for p in calibration_points) / len(calibration_points)
    weight_estimate = (raw_value - offset) / avg_scale
    
    # Iterative refinement
    for _ in range(iterations):
        # Get interpolated scale for current estimate
        scale = interpolate_scale(weight_estimate, calibration_points)
        
        # Recalculate weight with interpolated scale
        weight_estimate = (raw_value - offset) / scale
    
    return weight_estimate


def get_default_calibration():
    """
    Get default calibration values for fallback/testing
    
    Returns:
        dict: Default calibration with multi-point data
    """
    return {
        'type': 'multi_point',
        'offset': -217106,
        'points': [
            {'weight': 500, 'scale': -113.03},
            {'weight': 1000, 'scale': -112.89},
            {'weight': 1500, 'scale': -112.75},
            {'weight': 2000, 'scale': -112.60}
        ]
    }


# Example usage:
if __name__ == '__main__':
    # Test interpolation
    points = [
        {'weight': 500, 'scale': -113.03},
        {'weight': 1000, 'scale': -112.89},
        {'weight': 1500, 'scale': -112.75},
        {'weight': 2000, 'scale': -112.60}
    ]
    
    # Test weight at 1200g (between 1000 and 1500)
    raw = -354000
    offset = -217106
    
    weight = calculate_weight_multipoint(raw, offset, points)
    print(f"Weight: {weight:.1f}g")
    
    # Expected: interpolate between 1000g and 1500g calibrations
    # Should be more accurate than using single 500g calibration
