import face_recognition
import cv2
import numpy as np
import sys
import os.path
from os import path

try:
    if (path.exists(sys.argv[1])):
        print(sys.argv[1])
        person_image = face_recognition.load_image_file(sys.argv[1])
        person_name = sys.argv[2]
        person_face_encoding = face_recognition.face_encodings(person_image)[0]
        print("Person encoding:", person_face_encoding.shape)
except Exception as e:
    print("No system argument provided, please enter an image file name, followed by person name")
    quit()
    raise

if(path.exists('encoding.npy') and path.exists('names.npy')):
    original_arr_encoding = np.load('encoding.npy')
    original_arr_encoding = original_arr_encoding.tolist()
    person_face_encoding = person_face_encoding.tolist()
    original_arr_encoding.append(person_face_encoding)

    # Debug Statement
    # print(original_arr_encoding)

    original_arr_names = np.load('names.npy')
    original_arr_names = np.append(original_arr_names, [person_name], axis=0)
    np.save('encoding', original_arr_encoding)
    np.save('names', original_arr_names)


else:
    known_face_encodings = [person_face_encoding]
    known_face_names = [person_name]
    np.save('encoding', known_face_encodings)
    np.save('names', known_face_names)

    # Debug Statement
    # print("no existing file, encodings", known_face_encodings)
    # print("no existing file, names", known_face_names)
    # print("Original arr length if file dont exist:", len(known_face_encodings))
