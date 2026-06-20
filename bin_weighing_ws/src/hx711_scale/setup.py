from setuptools import setup
import os
from glob import glob

setup(
    name='hx711_scale',
    version='0.0.1',
    packages=['hx711_scale'],
    install_requires=['setuptools'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/hx711_scale']),          # fixes marker warning
        ('share/hx711_scale', ['package.xml']), # fixes package.xml warning
    ],
    entry_points={
        'console_scripts': [
            'calibration_node = hx711_scale.calibration_node:main',
            'scale_node = hx711_scale.scale_node:main',
        ],
    },
)
