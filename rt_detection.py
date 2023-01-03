import cv2 as cv
import time, torch
from torch import hub
import numpy as np

def predestrians_detect():
    # initialize the HOG descriptor/person detector
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    cv.startWindowThread()

    # open webcam video stream
    cap = cv.VideoCapture(0)

    # the output will be written to output.avi
    out = cv.VideoWriter(
        'output.avi',
        cv.VideoWriter_fourcc(*'MJPG'),
        15.,
        (640,480))

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # resizing for faster detection
        frame = cv.resize(frame, (640, 480))
        # using a greyscale picture, also for faster detection
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        for (xA, yA, xB, yB) in boxes:
            # display the detected boxes in the colour picture
            cv.rectangle(frame, (xA, yA), (xB, yB),
                            (0, 255, 0), 2)
        
        # Write the output video 
        out.write(frame.astype('uint8'))
        # Display the resulting frame
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    # and release the output
    out.release()
    # finally, close the window
    cv.destroyAllWindows()
    cv.waitKey(1)

# predestrians_detect()

def simple_video_stream():
    cv.startWindowThread()
    cap = cv.VideoCapture(0)

    while(True):
        # reading the frame
        ret, frame = cap.read()
        # turn to greyscale:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # # apply threshold. all pixels with a level larger than 80 are shown in white. the others are shown in black:
        # ret,frame = cv.threshold(frame,80,255,cv.THRESH_BINARY)
        # displaying the frame
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            # breaking the loop if the user types q
            # note that the video window must be highlighted!
            break

    cap.release()
    cv.destroyAllWindows()
    # the following is necessary on the mac,
    # maybe not on other platforms:
    cv.waitKey(1)

def get_video_stream():
    cap = cv.VideoCapture(0)
    model = hub.load( \
                      'ultralytics/yolov5', \
                      'yolov5s', \
                      pretrained=True)
"""
The function below identifies the device which is availabe to make the prediction and uses it to load and infer the frame. Once it has results it will extract the labels and cordinates(Along with scores) for each object detected in the frame.
"""
def score_frame(self, frame, model):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    frame = [torch.tensor(frame)]
    results = self.model(frame)
    labels = results.xyxyn[0][:, -1].numpy()
    cord = results.xyxyn[0][:, :-1].numpy()
    return labels, cord

"""
The function below takes the results and the frame as input and plots boxes over all the objects which have a score higer than our threshold.
"""
def plot_boxes(self, results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        # If score is less than 0.2 we avoid making a prediction.
        if row[4] < 0.2: 
            continue
        x1 = int(row[0]*x_shape)
        y1 = int(row[1]*y_shape)
        x2 = int(row[2]*x_shape)
        y2 = int(row[3]*y_shape)
        bgr = (0, 255, 0) # color of the box
        classes = self.model.names # Get the name of label index
        label_font = cv.FONT_HERSHEY_SIMPLEX #Font for the label.
        cv.rectangle(frame, \
                      (x1, y1), (x2, y2), \
                       bgr, 2) #Plot the boxes
        cv.putText(frame,\
                    classes[labels[i]], \
                    (x1, y1), \
                    label_font, 0.9, bgr, 2) #Put a label over box.
        return frame

"""
The Function below oracestrates the entire operation and performs the real-time parsing for video stream.
"""
def __call__(self):
    model = torch.hub.load( \
                      'ultralytics/yolov5', \
                      'yolov5s', \
                      pretrained=True)

    cap = cv.VideoCapture(0)
    player = self.get_video_stream() #Get your video stream.
    assert cap.isOpened() # Make sure that their is a stream. 
    #Below code creates a new video writer object to write our
    #output stream.
    x_shape = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    y_shape = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    size = (x_shape, y_shape)
    four_cc = cv.VideoWriter_fourcc(*"mp4v") #Using MJPEG codex
    out = cv.VideoWriter("out_file.mp4", four_cc, 20, size) 
    ret, frame = cap.read() # Read the first frame.
    while ret: # Run until stream is out of frames
        start_time = time() # We would like to measure the FPS.
        results = self.score_frame(frame, model) # Score the Frame
        frame = self.plot_boxes(results, frame) # Plot the boxes.
        end_time = time()
        fps = 1/np.round(end_time - start_time, 3) #Measure the FPS.
        print(f"Frames Per Second : {fps}")
        out.write(frame) # Write the frame onto the output.
        ret, frame = cap.read() # Read next frame.
    

    # cap = cv.VideoCapture(0)
    # # Define the codec and create VideoWriter object
    # width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) + 0.5)
    # height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) + 0.5)
    # size = (width, height)

    # fourcc = cv.VideoWriter_fourcc(*'mp4v')
    # out = cv.VideoWriter('output.mp4', fourcc, 20.0, size)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Can't receive frame (stream end?). Exiting ...")
    #         break
    #     # frame = cv.flip(frame, 0)
    #     start_time = time() # We would like to measure the FPS.
    #     results = self.score_frame(frame) # Score the Frame
    #     frame = self.plot_boxes(results, frame) # Plot the boxes.

    #     # # write the flipped frame
    #     # out.write(frame)
    #     # cv.imshow('frame', frame)
    #     # if cv.waitKey(1) == ord('q'):
    #     #     break
    # # Release everything if job is finished
    # cap.release()
    # out.release()
    # cv.destroyAllWindows()