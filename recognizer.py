import cv2
import face_recognition
import numpy as np
import os

known_faces_dir = "known_faces"

# Lists to store face encodings and names
known_face_encodings = []
known_face_names = []

# Loop through subfolders (each person's folder)
for person_name in os.listdir(known_faces_dir):
    person_folder = os.path.join(known_faces_dir, person_name)

    # Ensure it's a directory (skip any files accidentally placed in `known_faces/`)
    if os.path.isdir(person_folder):
        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)

            # Load image and encode face
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:  # Ensure at least one face encoding was found
                for encoding in encodings:
                    known_face_encodings.append(encoding)
                    known_face_names.append(person_name)  # Use the folder name as the person's name

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Resize frame to speed up processing (smaller image = faster processing)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # 50% of original size

    # Convert frame to RGB (face_recognition requires RGB images)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale back face locations to original frame size
        top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2

        name = "Unknown"
        color = (0, 0, 255)  # Default: Red for unknown faces

        # Compare detected face with known faces
        if known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

            if best_match_index is not None and matches[best_match_index]:
                name = known_face_names[best_match_index]
                color = (0, 255, 0)  # Green for recognized faces

        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Display name below face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

    # Show the frame
    cv2.imshow("Facial Recognition", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()