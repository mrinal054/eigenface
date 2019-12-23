import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

N = 10                                              # no. of features
imh = 144                                           # image height
imw = 108                                           # image width
pixels = imh*imw                                    # size of each image
im_class = 5                                        # no. of persons
im_per_folder = 5                                   # no. of images per person
im_total = im_class*im_per_folder                   # total no. of images

# Display some sample images
plt.figure()
for i in range(1, 5):
    im = cv2.imread('s' + str(i) + '\\' + '1.jpg', -1)                       # set image directory
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)                                  # convert BGR to RGB
    plt.subplot(2, 2, i)
    plt.imshow(im)
    plt.axis('off')
    plt.title('Sample ' + str(i))

# Make data frame to store images
df = pd.DataFrame(columns=range(0, pixels))

# Store the training images in 1D format
val = 0
for k in range(0, im_class):
    for i in range(0, im_per_folder):
        # Read and store image
        img = cv2.imread('s' + str(k+1) + '\\' + str(i+1) + '.jpg', 0)          # set image directory
        arr = np.array(img)                                                     # store in array
        flat_arr = arr.ravel()                                                  # convert to 1d
        df.loc[val] = flat_arr                                                  # stored in row-wise
        val = val + 1

# Create image matrix and label
w = np.transpose(df)                                                # images are now column-wise (15552x25)
print(('Size of image matrix: ' + str(np.shape(w))))
print('Image matrix:\n' + str(w))

# Calculate mean image
m = np.mean(w, axis=1)                                              # row-wise mean (15552x1)
mean = np.array(m)
mean = np.reshape(mean , (-1, 1))
print('size of mean: ' + str(np.shape(mean)))

# Show mean image
meanIm = np.reshape(mean, (imh, imw))
plt.figure()
plt.imshow(meanIm)
plt.axis('off')
plt.title('Mean image')

# Create ones
O = np.ones((1, im_total), dtype=int)                               # 1x25

# Substract w from the mean image
repMean = np.matmul(mean, O)                                        # replicating mean
vzm = np.subtract(w, repMean)
vzm = np.array(vzm)
vzm = vzm.astype('int')

print('size of vzm: ' + str(np.shape(vzm)))

# Square matrix
L = np.matmul(np.transpose(vzm), vzm)

# Calculate eigenvalue and eigenvector
eigvalue, eigvec = np.linalg.eig(L)
print('Size of eigvalue: ' + str(np.shape(eigvalue)) + '\n' + 'Size of eigvec: ' + str(np.shape(eigvec)))
print('eigvlaue:\n' + str(eigvalue))

# Sort eigenvalues
B = np.sort(eigvalue)[::-1]                                     # sorting in descending order
print('Print B:\n' + str(B))

index = np.argsort(np.array(eigvalue))[::-1]                    # index for descending order
print('index:\n' + str(index))

# Plot eigenvalues
xVal = list(range(im_total))
plt.figure()
plt.plot(xVal, B)
plt.xlabel('No. of eigenvalue')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalue plotting')

# Plot variance
cVal = np.zeros(shape=(1, im_total))                            # cVal = cumulative value
cVal[0, 0] = B[0]

print('iteration: ' + str(0) + '\t' + 'cVal: ' + str(cVal[0,0]) + '\t\t' + 'B: ' + str(B[0]))

for i in range(1, im_total):
    cVal[0, i] = np.add(cVal[0, i-1], B[i])
    print('iteration: ' + str(i) + '\t' + 'cVal: ' + str(cVal[0,i]) + '\t\t' + 'B: ' + str(B[i]))

plt.figure()
xVal2 = np.reshape(xVal, (1, im_total))
plt.plot(xVal2, cVal, 'ro')
plt.xlabel('Iteration')
plt.ylabel('Variance')
plt.axis([-1, im_total, 0, (np.max(cVal) + 100)])
plt.title('Variance plotting')

# Sort eigenvectors according to the eigenvalues
eigvec_sorted = np.zeros(shape=(im_total, im_total))

for i in range(0, im_total):
    eigvec_sorted[:,i] = eigvec[:,index[i]]

#print('eigvec_sorted:\n' + str(eigvec_sorted))
chosen_vec = eigvec_sorted[:,0:N]                                           # select no. of features
print('Size of chosen_vec: ' + str(np.shape(chosen_vec)))
print('chosen_vec:\n' + str(chosen_vec))

# Create feature matrix
feature_matrix = np.matmul(vzm,chosen_vec)
print('Size of feature_matrix: ' + str(np.shape(feature_matrix)))           # size: pixels x N

# Create eigenface
eigenface = np.zeros(shape=(imh, imw, N))
for i in range(0, N):
    eigf = feature_matrix[:, i]
    eigf_reshape = np.reshape(eigf, (imh, imw))
    eigenface[:, :, i] = eigf_reshape

print('Size of eigenface: ' + str(np.shape(eigenface)))

# Display top ten eigenfaces
plt.figure(figsize=(8,8))
for i in range(0, N):
    plt.subplot(4, 3, i+1)
    plt.imshow(eigenface[:, :, i], interpolation='bicubic')
    plt.axis('off')
    plt.title('Eigenface ' + str(i+1))

plt.show()