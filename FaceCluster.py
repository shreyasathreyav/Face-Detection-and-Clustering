import numpy as np
import cv2
import os
import json
import face_recognition
import sys

img_path = ""

cmd_args = sys.argv

for i in range(1, len(cmd_args)):

    if i == 1:
        img_path = img_path + cmd_args[i]

    else:
        img_path = img_path + " " + cmd_args[i]

K = int(img_path[-1])

def face_detection(imgPath, imageName, json_list,X, file_list, roi_faces):
    
    det_model = "./Model_Files/res10_300x300_ssd_iter_140000.caffemodel"
    det_config = "./Model_Files/deploy.prototxt.txt"
    
    net = cv2.dnn.readNetFromCaffe(det_config, det_model)
    
    img_read = cv2.imread(imgPath)
    og_img = img_read.copy()
    img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)

    (h, w) = img_read.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(img_read, (250, 250)), 1.0, (224, 224))

    net.setInput(blob)
    detect_results = net.forward()
    
    # loop over the detections
    for i in range(0, detect_results.shape[2]):

        detection_val = detect_results[0, 0, i, 2]
        
        detection_threshold = 0.5

        if detection_val > detection_threshold:
            
            bounding_box = detect_results[0, 0, i, 3:7] * np.array([w, h, w, h])
            
            (x, y, x1, y1) = bounding_box.astype("int")
            
            boxes = [(y,x1,y1,x)]
            
            encoding_val = face_recognition.face_encodings(img_read, boxes)
            
            X.append(np.array(encoding_val))
            
            file_list.append(imageName)
            
            roi_faces.append(cv2.resize(og_img[y:y1, x:x1], (100, 100)))

            
json_list = []
file_list =[]
X = []
roi_faces =[]
for imageName in os.listdir(img_path):
    imgPath = os.path.join(img_path, imageName)
    face_detection(imgPath, imageName, json_list, X, file_list, roi_faces)

X = np.array(np.reshape(X, (len(X), 128)))
num_epochs = 100

def SSD_cal(img1, img2 ):
    
    error = (img1 - img2)**2
    
    sum_error = np.sum(error) 
    
    return np.sqrt(sum_error)

def centroids_initialization(X, K):
    """
code for initializing the cetroids by finding the maximum distance between centroid points  

    """
    num_imgs = X.shape[0]
    
    list_centroids = []
    
    random_val = np.random.randint(num_imgs)
    
    list_centroids.append(X[random_val, :])

    for clust_num in range(K - 1):
          
        distance_list = []
        
        for img_idx in range(num_imgs):
            
            img_point = X[img_idx, :]
            
            max_dist = np.Inf
              
            for each_centroid in list_centroids:
                
                new_dist = SSD_cal(img_point, each_centroid)
                
                max_dist = min(max_dist, new_dist)
                
            distance_list.append(max_dist)
              
        distance_list = np.array(distance_list)
        
        max_point = np.argmax(distance_list)
        
        new_centroid = X[max_point, :]
        list_centroids.append(new_centroid)
        
    return list_centroids

def cluster_initialization(centroids, X):

    global K
    
    cluster_list = []
    
    for i in range(K):
        l = []
        cluster_list.append(l)
    
    for indx, sam_val in enumerate(X):
        
        ssd_list = [SSD_cal(sam_val, point) for point in centroids]
        centroid_indices = np.argmin(ssd_list)
        cluster_list[centroid_indices].append(indx)
    return cluster_list

def centroid_cal(clusters, X):
    
    num_imgs, ft_len = X.shape
    
    global K
    
    new_centroids = np.zeros((K, ft_len))
    
    for indx, clust in enumerate(clusters):
        
        cluster_mean = np.mean(X[clust], axis=0)
        
        new_centroids[indx] = cluster_mean
        
    return new_centroids

def final_clust_val(clusters, X):
    
    n_sam, n_ft = X.shape

    final_val = np.empty(n_sam)

    for clust_index, cluster_val in enumerate(clusters):
        for i in cluster_val:
            final_val[i] = clust_index
    return final_val


def clustering(X):
    
    global K

    global num_epochs
    
    n_sam, n_ft = X.shape

    centroid_list = centroids_initialization(X, K)

    for iter in range(num_epochs):

        cluster_list = cluster_initialization(centroid_list, X)
        
        centroid_list_temp = centroid_list
        current_centroid = centroid_cal(cluster_list, X)
    
        dist_list = []
            
        for i in range(K):
            temp = SSD_cal(centroid_list_temp[i], current_centroid[i])
            dist_list.append(temp)
            
        check = sum(dist_list)==0
    
        if check:
            break

    return final_clust_val(cluster_list, X)


cluster_result = clustering(X)


output_list = []
img_cluster = {}
for result in range(len(cluster_result)):
    
    
    val = cluster_result[result]
    if val not in img_cluster.keys():
        img_cluster[val] = np.array(roi_faces[result])
    else:
        img_cluster[val] = np.concatenate((img_cluster[val], roi_faces[result]), axis = 1)
        
#count = 0
#for cluster in img_cluster.values():
    
    # cv2.imshow("Clusters Image", clusters)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #cv2.imwrite("cluster_{}.jpg".format(count), cluster)
    #count +=1

for i in range(K):
    indices = np.where(cluster_result == i )[0]
    cluster_list =[]
    for j in indices:
        cluster_list.append(file_list[j])
    val = {"cluster_no": i, "elements":cluster_list}
    output_list.append(val)

json_name = "clusters.json"
with open(json_name,'w') as f:
    json.dump(output_list, f)


#####################################
"""
References

1) https://www.geeksforgeeks.org/ml-k-means-algorithm/

2) https://www.youtube.com/watch?v=vtuH4VRq1AU

3) https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/kmeans.py

"""


