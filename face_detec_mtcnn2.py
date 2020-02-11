import face_recognition
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from os import path
import time

detector = MTCNN()

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(1)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    # small_frame = frame
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    np.load.__defaults__ = (None, True, True, 'ASCII')

    while not path.exists('encoding.npy'):
        time.sleep(1)
    while True:
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
        if len(known_face_encodings) == len(known_face_names):
            break
        else:
            continue

# print(known_face_encodings.shape)
# print(known_face_names)

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
                      (bounding_box[0] * 4, bounding_box[1] * 4),
                      (bounding_box[0] * 4 + bounding_box[2] * 4, bounding_box[1] * 4 + bounding_box[3] * 4),
                      (0, 155, 255),
                      2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # cv2.imshow('SMALL_FRAME', small_frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
