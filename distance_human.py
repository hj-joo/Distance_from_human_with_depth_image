
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils import ROOT, yaml_load

import json
from time import time



# parameters
WITDH = 1280
HEIGHT = 720
model = YOLO('yolov8n.pt')
CLASSES = yaml_load(check_yaml('coco128.yaml'))['names']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))



# Create a config and configure the pipeline to stream for Realsense-L515
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


# Start streaming
profile = pipeline.start(config)

# # Getting the depth sensor's depth scale (see rs-align example for explanation)
# depth_sensor = profile.get_device().first_depth_sensor()
# depth_scale = depth_sensor.get_depth_scale()
# # print("Depth Scale is: " , depth_scale)

# # We will be removing the background of objects more than
# #  clipping_distance_in_meters meters away
# clipping_distance_in_meters = 1 #1 meter
# clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())


        depth_image = cv2.resize(depth_image, (WITDH, HEIGHT))
        color_image = cv2.resize(color_image, (WITDH, HEIGHT))

        # # Remove background - Set pixels further than clipping_distance to grey
        # grey_color = 153
        # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # # Render images:
        # #   depth align to color on left
        # #   depth on right
        depth_image_scaled = cv2.convertScaleAbs(depth_image, alpha=0.025)
        results= model(color_image, stream=True)

        class_ids = []
        confidences = []
        bboxes = []
        obj_centers = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf
                if confidence > 0.5:
                    xyxy = box.xyxy.tolist()[0]
                    bboxes.append(xyxy)
                    confidences.append(float(confidence))
                    class_ids.append(box.cls.tolist())
                    cx = int((xyxy[2]+xyxy[0])//2)
                    cy = int((xyxy[3]+xyxy[1])//2)
                    obj_centers.append([cx,cy])

        result_boxes = cv2.dnn.NMSBoxes(bboxes, confidences, 0.25, 0.45, 0.5)

        # print(result_boxes)
        font = cv2.FONT_HERSHEY_PLAIN
        depth_list = list()
        person_id_list = list()
        for i in range(len(bboxes)):

            label = str(CLASSES[int(class_ids[i][0])])
            if label == 'person':
                if i in result_boxes:
                    bbox = list(map(int, bboxes[i])) 
                    x, y, x2, y2 = bbox
                    cx, cy = obj_centers[i]
                    depth = depth_image_scaled[cy, cx]

                    depth_list.append(depth)  
                    person_id_list.append(i)
                    start = time()
                    object_data = list()
                    # print(len(person_id_list))


                    # json file format
                    for i in range(len(person_id_list)):
                        json_data = {
                            "id": person_id_list[i],
                            "pose3d" : depth_list[i].tolist(), # Distance from object
                            "available" : True,
                            "time" : start
                        }
                        object_data.append(json_data)
                        #print(json_data)
                    list_data = {
                            "key": object_data
                    }
                    # print(list_data)


                    color = colors[i]
                    color = (int(color[0]), int(color[1]), int(color[2]))

                    cv2.rectangle(color_image, (x, y), (x2, y2), color, 2)
                    cv2.circle(depth_image_scaled,(cx,cy),20,color,20)
                    cv2.rectangle(depth_image_scaled, (x, y), (x2, y2), color, 2)
                    cv2.putText(color_image, "{} cm".format(depth), (x + 5, y + 60), 0, 1.0, color, 2)
                    cv2.putText(depth_image_scaled, "{} cm".format(depth), (x + 5, y + 60), 0, 1.0, color, 2)
                    cv2.putText(color_image, label, (x, y + 30), font, 3, color, 3)

        cv2.imshow("Bgr frame", color_image)
        cv2.imshow("Depth frame", depth_image_scaled)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()



