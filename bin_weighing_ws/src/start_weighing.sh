#!/bin/bash
# Start Bin Weighing System (2-program architecture)
# Launches scale + cameras, then backend + manager
# Auto-tares scale on startup

echo "Starting Bin Weighing System..."
echo "Note: This requires sudo for GPIO access"
echo ""

# Start scale + cameras first (individually, not the old launch file)
echo "Starting scale node..."
sudo bash -c "source /opt/ros/jazzy/setup.bash && \
              source /home/group5/bin_weighing_ws/install/setup.bash && \
              ros2 run hx711_scale scale_node" > /tmp/scale_node.log 2>&1 &
SCALE_PID=$!

echo "Starting cameras..."
sudo bash -c "source /opt/ros/jazzy/setup.bash && \
              source /home/group5/bin_weighing_ws/install/setup.bash && \
              ros2 run camera_ros camera_node" > /tmp/camera_node.log 2>&1 &
CAMERA_PID=$!

sudo bash -c "source /opt/ros/jazzy/setup.bash && \
              source /home/group5/bin_weighing_ws/install/setup.bash && \
              ros2 run gv_ros2 gv_ros2_node" > /tmp/depth_camera.log 2>&1 &
DEPTH_PID=$!

# Wait for nodes to initialize
echo "Waiting for nodes to initialize..."
sleep 5
echo ""

# Start backend in background with sudo (redirect logs)
echo "Starting backend (will run in background)..."
sudo bash -c "source /opt/ros/jazzy/setup.bash && \
              source /home/group5/bin_weighing_ws/install/setup.bash && \
              python3 $(pwd)/bin_weighing_backend.py" > /tmp/bin_weighing_backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to initialize
sleep 3

# Start manager (user interface) in foreground with sudo
echo "Starting user interface..."
sudo bash -c "source /opt/ros/jazzy/setup.bash && \
              source /home/group5/bin_weighing_ws/install/setup.bash && \
              python3 $(pwd)/bin_weighing_manager.py"

# Cleanup when manager exits
echo ""
echo "Shutting down..."
sudo kill $BACKEND_PID 2>/dev/null
sudo kill $SCALE_PID 2>/dev/null
sudo kill $CAMERA_PID 2>/dev/null
sudo kill $DEPTH_PID 2>/dev/null

echo "System stopped."
