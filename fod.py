import face_recognition
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from os import path
import time

# ==========================================<--MTCNN-->=========================================================================

detector = MTCNN()

# Get a reference to webcam #0 (the default one)
# video_capture = cv2.VideoCapture(1)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# ==========================================<--YOLO-->=========================================================================
# Load Yolo
net = cv2.dnn.readNet("yolov3-tiny-obj_last.weights", "yolov3-tiny-obj.cfg")
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
outputLayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

# ==========================================<--DETECTION-->=========================================================================

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    #small_frame = frame
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    height, width, channels = small_frame.shape

    #YOLO
    yolo_blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False) 
    net.setInput(yolo_blob)
    yolo_outs = net.forward(outputLayers)

    # Showing info on screen + get confidence score of algorithm in detecting an object in blob
    class_ids = []
    confidences = []
    boxes = []

    for out in yolo_outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3: #object detected!
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

    for i in range (len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes(class_ids[i]))
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    np.load.__defaults__=(None, True, True, 'ASCII')
    
    while not path.exists('encoding.npy'):
        time.sleep(1)
    while True:
        try:
            known_face_encodings = np.load('encoding.npy')
            break
        except:
            continue
    
    while not path.exists('names.npy'):
        time.sleep(1)
    while True:
        try:
            known_face_names = np.load('names.npy')
            break
        except:
            continue
   #print(known_face_encodings.shape)
    #print(known_face_names)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        result_mtcnn = detector.detect_faces(rgb_small_frame)
        #print("MTCNN", result_mtcnn["box"])

        fl = []
        for face in result_mtcnn:
            bb = face['box']
            fl.append((bb[1], bb[0]+bb[2], bb[1]+bb[3], bb[0]))
        #print("MTCNN", fl)
        face_locations = fl
        #face_locations = face_recognition.face_locations(rgb_small_frame)
        #print("Face_Locs", type(face_locations))
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        bounding_box = result_mtcnn[0]['box']
        cv2.rectangle(frame,
                          (bounding_box[0]*4, bounding_box[1]*4),
                          (bounding_box[0]*4+bounding_box[2]*4, bounding_box[1]*4 + bounding_box[3]*4),
                          (0,155,255),
                          2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    #cv2.imshow('SMALL_FRAME', small_frame)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
