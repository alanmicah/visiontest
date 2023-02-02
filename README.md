# Hydrating Recognition

Locate and demarcate hand gestures and drinking containers such as mugs, cups and water bottles, in live video using different vision and machine learning frameworks.

## Overview

Using [Mediapipe](https://google.github.io/mediapipe/) to perceive the shape of hands. This is done using Mediapipe's firstly the Palm Detection Model to detect the intial location of each hand and then using the Hand LandMark Model to find localise 21 precise keypoints of hand and knuckle coordinates.

Also using [OpenCV](https://opencv.org/) to maintain the video feed and on screen drawings and text.

Currently the video feed will display text on the screen to indicate when a drinking motion is detected, this is only using a fixed range of custom lower and upper coordiante boundaries.

## Next Steps

To detect a broader range of acceptable hand gestures to be detected as drinking, be able to recognise the hand gesture and detect if a drinking object is in frame and finally whether or not a face is in frame.

A specifically trained model that can recognise all these motions together as drinking would be ideal.
