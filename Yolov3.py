import cv2
import numpy as np
import urllib.request
import requests

# Load YOLOv3 config and weights
yolo_net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

# Telegram bot API token and URL
apiToken = '6049697058:AAFEpdCheN0bq9mbwva9Q0Op9Rb1twWTlLA'
chatID = '772942694'
apiURL = f'https://api.telegram.org/bot{apiToken}/sendPhoto'

# Load COCO class labels
with open("coco.names", "r") as file:
    classes = file.read().strip().split("\n")
    
# Replace 'your_esp32_camera_ip_address' with the actual IP address of your ESP32 camera
camera_url = 'http://192.168.254.27/cam-mid.jpg'


# Open a connection to your laptop's camera (usually the built-in webcam)
# pylint: disable=no-member
cap = cv2.VideoCapture(camera_url)

while True:
    # Fetch frame from the ESP32 camera
    stream = urllib.request.urlopen(camera_url)
    bytes_data = bytes()
    while True:
        bytes_data += stream.read(1024)
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_data[a:b + 2]
            bytes_data = bytes_data[b + 2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            break

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set input to the network
    yolo_net.setInput(blob)

    # Get output layer names
    output_layer_names = yolo_net.getUnconnectedOutLayersNames()

    # Forward pass to get object detections
    detections = yolo_net.forward(output_layer_names)

    # Initialize a flag to indicate if a person has been detected
    person_detected = False
    send_full_frame = False

    # Loop through the detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == "person" and not person_detected:
                # Extract the coordinates of the bounding box
                box = obj[0:4] * np.array([width, height, width, height])
                (center_x, center_y, box_width, box_height) = box.astype("int")

                # Calculate the top-left corner of the bounding box
                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))

                # Draw a rectangle around the person
                cv2.rectangle(frame, (x, y), (x + int(box_width), y + int(box_height)), (0, 255, 0), 2)

                # Set the flag to indicate that a person has been detected
                person_detected = True
                
    if person_detected:
        send_full_frame = True

    # Display the frame with person detection
    cv2.imshow("Person Detection", frame)
    
    # If a person is detected, crop the image and send it to Telegram
    if send_full_frame:
        # Crop the detected person
        #cropped_image = frame[y:y + int(box_height), x:x + int(box_width)]
        
        cropped_image_resized = cv2.resize(frame, (800, 600))
        
        # Save the cropped image to a file (temporary step)
        cv2.imwrite('resized_cropped_person.jpg', cropped_image_resized)

        # Read the saved image file
        with open('resized_cropped_person.jpg', 'rb') as img:
            files = {'photo': img}
            
            # Send the cropped image to the Telegram bot
            requests.post(apiURL, data={'chat_id': chatID}, files=files)
            
        # Optional: Delete the temporary file
        import os
        os.remove('resized_cropped_person.jpg')
    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
