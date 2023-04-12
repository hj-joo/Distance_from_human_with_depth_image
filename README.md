# Distance_from_human_with_depth_image
We get distance from human with realsense depth camera using human detection with yolov8 and we send the results to server.


## 1. Install Realsense Library in Ubuntu 20.04
```
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
export http_proxy="http://<proxy>:<port>"
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
sudo apt-get install librealsense2-dkms
sudo apt-get install librealsense2-utils
sudo apt-get install librealsense2-dev
sudo apt-get install librealsense2-dbg
```

## 2. Installation
```
git clone "https://github.com/hj-joo/Distance_from_human_with_depth_image.git"
pip install -r requirements.txt  # install dependencies
```

## 3. Get distance from human & Send to server
```
python distance_human.py
```

## 4. Result
![Screenshot from 2023-04-12 14-21-44](https://user-images.githubusercontent.com/88313282/231365007-4a091a6b-d595-4f9f-acb6-1667a22241cc.png)

