import  cv2
import  numpy as np
import  face_recognition
import  os
from    datetime import datetime
# Get the path of the images folder.
# import all images from ImagesAttendance folder
path = 'Attendance_Images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
namesSeen = []
namesNotSeen = []

# Loop through all images in the folder and append them to the images list.
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

# Function to find the encodings of all images in the images list and add them to a list.
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Function to mark the attendance of the person in the attendance.csv file.
def markAttendance(mName):
    with open('Attendance_List.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        #print(myDataList)
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if mName not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            dayandmonthyear = now.strftime('%D/%M/%Y')
            #If before 9:15am, mark as present
            if dtString < '09:15:00':
                f.writelines(f'\n{mName},Present, {dtString}, {dayandmonthyear}')
                #elseif after 9:15am, mark as late
            elif dtString > '09:15:00':
                f.writelines(f'\n{mName},Late, {dtString}, {dayandmonthyear}')
                # elseif after 9:45am, mark as absent
            elif dtString > '09:45:00':
                f.writelines(f'\n{mName},Absent, {dtString}, {dayandmonthyear}')

def markAttendanceNotSeen(nName):
    with open('Attendance_List.csv','r+') as f:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            dayandmonthyear = now.strftime('%D/%M/%Y')
            print('names not seen', nName)
            # get the list of names in nameNotSeen and write them to the attendance.csv file marked as absent
            for Xname in nName:
                f.writelines(f'\n{Xname},Absent, {dtString}, {dayandmonthyear}')

encodeListKnown = findEncodings(images)
# print 'encoding complete and list length.
print('Encoding Complete', len(encodeListKnown))

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
            # add name to namesSeen list
            namesSeen.append(name)

    # if q key is pressed, break out of loop, mark attendance for all names not seen
    # and close the camera, all windows and the program.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('names seen: ', namesSeen)
        print('class names: ', classNames)
        namesNotSeen = (set(classNames) - set(namesSeen))
        markAttendanceNotSeen(namesNotSeen)
        # close the camera, all windows and the program.
        cap.release()
        cv2.destroyAllWindows()
        break

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)