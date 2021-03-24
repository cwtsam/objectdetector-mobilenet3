import cv2
import numpy as np

thres = 0.45 # threshold for detecting object
nms_threshold = 0.5
cap = cv2.VideoCapture(1) # accesses macbook webcam with value 1, pupil labs world cam: 3
cap.set(4,720) # height parameter
cap.set(10,150) # brightness

classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:
    success, img = cap.read() # img will be video frames
    classIds, confs, bbox = net.detect(img,confThreshold=thres) #if confidence is 50% and above, it's an object
    #print(classIds, bbox) #class ids refer back to the coco.names class names, bbox is bounding box

    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0]) # numpy array to list
    confs = list(map(float,confs)) # float32 to float
    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
        cv2.putText(img, classNames[classIds[i][0]-1].upper(), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        cv2.putText(img, str(round(confs[i] * 100, 2)), (x + w - 50, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) #outputs confidence score

    ## without NMS
    # if len(classIds) != 0:
    #     for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
    #         cv2.rectangle(img,box,color=(0,255,0),thickness=2) #draw green box for each detected object
    #         cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
    #                     cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2) #outputs detected object name
    #         cv2.putText(img, str(round(confidence*100,2)), (box[0] + 200, box[1] + 30),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #outputs confidence score

    cv2.imshow("Output",img)
    cv2.waitKey(1)

