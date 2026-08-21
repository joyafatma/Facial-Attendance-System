import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Set the folder containing student images
path = "images"

images = []
classNames = []

# Get all student subfolders
subfolders = [f.path for f in os.scandir(path) if f.is_dir()]

print("Total students detected:", len(subfolders))

# Read images from every student folder
for subfolder in subfolders:
    personName = os.path.basename(subfolder)

    # Get all files from the student folder
    imageFiles = os.listdir(subfolder)

    for imageFile in imageFiles:
        imagePath = os.path.join(subfolder, imageFile)

        # Read the image
        curImg = cv2.imread(imagePath)

        # Add only valid images
        if curImg is not None:
            images.append(curImg)
            classNames.append(personName)

# Create face encodings for known students
def findEncodings(images):
    encodeList = []

    for img in images:
        # Convert image from BGR to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Find faces in the image
        encodes = face_recognition.face_encodings(imgRGB)

        # Add encoding only if a face is found
        if len(encodes) > 0:
            encodeList.append(encodes[0])
        else:
            print("Warning: No face found in one image.")

    return encodeList


# Mark attendance in CSV file
def markAttendance(name):
    file_exists = os.path.isfile("Attendance.csv")

    # Open file in append and read mode
    with open("Attendance.csv", "a+", newline="") as f:
        # Move cursor to the beginning of the file
        f.seek(0)

        # Add header if the file is empty
        if not file_exists or os.path.getsize("Attendance.csv") == 0:
            f.write("Name,Date,Time\n")

        # Read existing attendance records
        myDataList = f.readlines()

        nameList = []

        # Get names already marked
        for line in myDataList:
            entry = line.strip().split(",")

            if len(entry) > 0:
                nameList.append(entry[0])

        # Mark attendance only once
        if name not in nameList:
            now = datetime.now()

            # Get current date
            dateString = now.strftime("%Y-%m-%d")

            # Get current time
            timeString = now.strftime("%H:%M:%S")

            # Write attendance record
            f.write(f"{name},{dateString},{timeString}\n")

            print(f"Attendance marked for: {name}")


# Generate known face encodings
encodeListKnown = findEncodings(images)

print("Encoding Complete")

# Check whether at least one face encoding exists
if len(encodeListKnown) == 0:
    print("Error: No valid face encodings found.")
    exit()

# Open Mac webcam
cap = cv2.VideoCapture(0)

# Check whether webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("Please allow camera access for VS Code in Mac settings.")
    exit()

print("Webcam started. Press ESC to exit.")

while True:
    # Read a frame from webcam
    success, img = cap.read()

    # Stop if webcam does not provide a frame
    if not success or img is None:
        print("Error: Could not read frame from webcam.")
        break

    # Resize frame for faster face recognition
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    # Convert webcam frame from BGR to RGB
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces in current frame
    facesCurFrame = face_recognition.face_locations(imgS)

    # Generate encodings for detected faces
    encodesCurFrame = face_recognition.face_encodings(
        imgS, facesCurFrame
    )

    # Compare every detected face with known faces
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

        # Compare current face with known faces
        matches = face_recognition.compare_faces(
            encodeListKnown, encodeFace
        )

        # Calculate face distances
        faceDis = face_recognition.face_distance(
            encodeListKnown, encodeFace
        )

        # Find the closest known face
        matchIndex = np.argmin(faceDis)

        # Check whether the closest face is a valid match
        if matches[matchIndex] and faceDis[matchIndex] < 0.6:
            name = classNames[matchIndex].upper()
        else:
            name = "UNKNOWN"

        # Get face coordinates
        y1, x2, y2, x1 = faceLoc

        # Scale coordinates back to original frame size
        y1 *= 4
        x2 *= 4
        y2 *= 4
        x1 *= 4

        # Draw rectangle around detected face
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (255, 0, 255),
            2
        )

        # Draw name background
        cv2.rectangle(
            img,
            (x1, y2 - 35),
            (x2, y2),
            (255, 0, 255),
            cv2.FILLED
        )

        # Display person's name
        cv2.putText(
            img,
            name,
            (x1 + 6, y2 - 6),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 0, 0),
            2
        )

        # Mark attendance for recognized student
        if name != "UNKNOWN":
            markAttendance(name)

    # Display webcam window
    cv2.imshow("Webcam", img)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()