import cv2

# Load the Haar Cascade algorithm
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

# Initialize the camera
cam = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Read a frame from the camera
    ret, img = cam.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert the image to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = haar_cascade.detectMultiScale(grayImg, scaleFactor=1.3, minNeighbors=4)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the result
    cv2.imshow("Face Detection", img)

    # Exit on pressing the 'Esc' key
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the camera and close the windows
cam.release()
cv2.destroyAllWindows()
