import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# # We define the histogram by describing its bins.
# # The bins are baskets that count the number of entries with a value falling within the bin range.
# values = [1.1, 1.5, 2.2, 3.5, 3.5, 3.6, 4.1]
# plt.hist(values, bins=4, range=(1,5))
# plt.show()

# Values' contribution to the bin is given by the weight
# values = [1.1, 1.5, 2.2, 3.5, 3.5, 3.6, 4.1]
# weights = [1., 1., 3., 1.2, 1.4, 1.1, 0.2]
# plt.hist(values, bins=4, range=(1,5), weights=weights)
# plt.show()

cell = np.array([
    [0, 1, 2, 5, 5, 5, 5, 5],
    [0, 0, 1, 4, 4, 5, 5, 5],
    [0, 0, 1, 3, 4, 5, 5, 5],
    [0, 0, 0, 1, 2, 3, 5, 5],
    [0, 0, 0, 0, 1, 2, 5, 5],
    [0, 0, 0, 0, 0, 1, 3, 5],
    [0, 0, 0, 0, 0, 0, 2, 5],
    [0, 0, 0, 0, 0, 0, 1, 3],
    ],dtype='float64')

# compute the gradients in the x and y directions
gradx = cv.Sobel(cell, cv.CV_64F,1,0,ksize=1)
grady = cv.Sobel(cell, cv.CV_64F,0,1,ksize=1)
# compute the magnitude and angle of the gradients
norm, angle = cv.cartToPolar(gradx,grady,angleInDegrees=True)

plt.figure(figsize=(10,5))

# display the image
plt.subplot(1,2,1)
plt.imshow(cell, cmap='binary', origin='lower')

# display the magnitude of the gradients:
plt.subplot(1,2,2)
plt.imshow(norm, cmap='binary', origin='lower')
# and superimpose an arrow showing the gradient
# magnitude and direction: 
q = plt.quiver(gradx, grady, color='blue')
plt.savefig('gradient.png')
plt.show()
# histogram of oriented gradients
plt.hist(angle.reshape(-1), weights=norm.reshape(-1), bins=20, range=(0,360))
plt.show()

