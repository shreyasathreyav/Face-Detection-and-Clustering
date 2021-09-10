import cv2
import numpy as np
import sys
import os
import json

det_model = "Model_Files/res10_300x300_ssd_iter_140000.caffemodel"
det_config = "Model_Files/deploy.prototxt.txt"


img_path = ""

cmd_args = sys.argv

for i in range(1, len(cmd_args)):
    if i == 1:
        img_path = img_path + cmd_args[i]
    else:
        img_path = img_path + " " + cmd_args[i]

img_path = os.path.join(img_path, "images")

detector = cv2.dnn.readNetFromCaffe(det_config, det_model)

json_name = "results.json"

results_json =[]

for name_img  in os.listdir(img_path):

    full_name = (img_path + "/" + name_img)
    
    image = cv2.imread(full_name)
    
    
    (ht, wd) = image.shape[:2]
    
    img_segmentation = cv2.dnn.blobFromImage(image, 1.2, (227, 227))
    
    detector.setInput(img_segmentation)
    
    detect_results = detector.forward()
    
    len_detected = detect_results.shape[2]
    
    
    detection_threshold = 0.6
    
    for i in range(len_detected):
    
        sample_val = detect_results[0, 0, i, 2]
        
        if sample_val > detection_threshold :
        
            detection = detect_results[0, 0, i, 3:7] * np.array([wd, ht, wd, ht])
            x_val, y_val, width, height = int(detection[0]), int(detection[1]), (int(detection[2]) - int(detection[0])), (int(detection[3]) - int(detection[1]))
            bb_box = {"iname" : name_img, "bbox" : [x_val, y_val, width, height]}
            results_json.append(bb_box)



with open(json_name, 'w') as f:
    json.dump(results_json, f)


################################################
"""
References

1) https://github.com/AyushExel/ml4face-detection

2) https://towardsdatascience.com/extracting-faces-using-opencv-face-detection-neural-network-475c5cd0c260

3) https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/#:~:text=Informally%2C%20a%20blob%20is%20just,preprocessed%20in%20the%20same%20manner.&text=cv2.dnn.blobFromImages-,cv2.,functions%20are%20near%20identical.

"""
