from cvzone.HandTrackingModule import HandDetector
import cv2

# detector = HandDetector(maxHands=1, detectionCon=0.8)
# video = cv2.VideoCapture(0)

# while True:
#     # # Get image frame
#     # success, img = video.read()
#     # # Find the hand and its landmarks
#     # hands, img = detector.findHands(img)  # with draw
#     # hands = detector.findHands(img, draw=False)  # without draw
    
#     _, img = video.read()
#     # img = cv2.flip(img, 1)
#     hand = detector.findHands(img, draw=False)
#     fing = cv2.imread("Put image path with 0 fingures up")
#     if hand:
#         lmlist = hand[0]
#         if lmlist:
#             fingerup = detector.fingersUp(lmlist)
#             if fingerup == [0, 1, 0, 0, 0]:
#                 fing = cv2.imread("Put image \
#                 path of 1 fingures up")
#             if fingerup == [0, 1, 1, 0, 0]:
#                 fing = cv2.imread("Put image \
#                 path of 2 fingures up")
#             if fingerup == [0, 1, 1, 1, 0]:
#                 fing = cv2.imread("Put image \
#                 path of 3 fingures up")
#             if fingerup == [0, 1, 1, 1, 1]:
#                 fing = cv2.imread("Put image \
#                 path of 4 fingures up")
#             if fingerup == [1, 1, 1, 1, 1]:
#                 fing = cv2.imread("Put image \
#                 path of 4 fingures and thumbs up")
#     fing = cv2.resize(fing, (220, 280))
#     img[50:330, 20:240] = fing
#     cv2.imshow("Video", img)
      
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
          
# video.release()
# cv2.destroyAllWindows()

####
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers1 = detector.fingersUp(hand1)
        print(fingers1)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"

            fingers2 = detector.fingersUp(hand2)

            # Find Distance between two Landmarks. Could be same hand or different hands
            length, info, img = detector.findDistance(lmList1[8][:2], lmList2[8][:2], img)  # with draw
            # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
####

# import cvzone
# import cv2
# # from cvzone.HandTrackingModule import HandDetector

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
# detector = cvzone.HandDetector(detectionCon=0.5, maxHands=1)

# while True:
#     # Get image frame
#     success, img = cap.read()

#     # Find the hand and its landmarks
#     img = detector.findHands(img)
#     lmList, bbox = detector.findPosition(img)

#     if lmList:
#         # Find how many fingers are up
#         fingers = detector.fingersUp()
#         totalFingers = fingers.count(1)
#         cv2.putText(img, f'Fingers:{totalFingers}', (bbox[0] + 200, bbox[1] - 30),
#                     cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

#     if lmList:
#         # Find Distance Between Two Fingers
#         distance, img, info = detector.findDistance(8, 12, img)
#         cv2.putText(img, f'Dist:{int(distance)}', (bbox[0] + 400, bbox[1] - 30),
#                     cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
#     # Display
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)
