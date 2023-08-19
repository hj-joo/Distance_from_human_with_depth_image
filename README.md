# Distance_from_human_with_depth_image
We get distance from human with realsense depth camera using human detection with yolov8 and we send the results to server.


## 1. Install Realsense Library in Ubuntu
```
sudo apt-get install -y git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev
sudo apt-get install -y libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev

git clone https://github.com/IntelRealSense/librealsense.git
cd ./librealsense
./scripts/setup_udev_rules.sh
mkdir build && cd build
cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true
sudo make uninstall && sudo make clean && sudo make -j4 && sudo make install
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.X/pyrealsense2
```

## 2. Installation
```
git clone "https://github.com/hj-joo/Distance_from_human_with_depth_image.git"
pip install -r requirements.txt  # install dependencies
```

## 3. Get distance from human
```
python distance_human.py
```

## 4. Result
![Screenshot from 2023-04-12 14-21-44](https://user-images.githubusercontent.com/88313282/231365007-4a091a6b-d595-4f9f-acb6-1667a22241cc.png)

