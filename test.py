import face_recognition as fr
import cv2 as cv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import matplotlib.pyplot as plt

Tk().withdraw()
loadImage = askopenfilename()
targetImage = fr.load_image_file(loadImage)
targetEncoding = fr.face_encodings(targetImage)[0]  # Extract the first encoding from the list

def encode_faces(folder):
    list_people_encoding = []
    for filename in os.listdir(folder):
        known_image = fr.load_image_file(os.path.join(folder, filename))  # Fix the path format
        known_encoding = fr.face_encodings(known_image)[0]  # Extract the first encoding from the list
        list_people_encoding.append((known_encoding, filename))
    
    return list_people_encoding

def find_target_face():
    face_Location = fr.face_locations(targetImage)
    num_known_faces = len(encode_faces('people/'))

    for Location in face_Location:
        label = None
        for person in encode_faces('people/'):
            encoded_face = person[0]
            filename = person[1]

            is_target_face = fr.compare_faces([encoded_face], targetEncoding, tolerance=0.55)
            if any(is_target_face):
                label = filename
                break

        if label is not None:
            create_frame(Location, label)

def create_frame(location, label):
    top, right, bottom, left = location

    # Draw a blue rectangle around the face
    cv.rectangle(targetImage, (left, top), (right, bottom), (255, 0, 0), 2)

    # Draw a blue background rectangle for the label text
    cv.rectangle(targetImage, (left, bottom - 20), (right, bottom), (255, 0, 0), cv.FILLED)

    # Set the label text
    label_text = f'Matched: {label}'

    # Put the label text on the image
    cv.putText(targetImage, label_text, (left + 2, bottom - 5), cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)


def render_image():
    rgb_img = cv.cvtColor(targetImage, cv.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.title('Face Recognition')
    plt.axis('off')
    plt.show()

find_target_face()
render_image()
