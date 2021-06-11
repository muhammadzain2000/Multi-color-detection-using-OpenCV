import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter


# image = cv2.imread('sample_image.jpeg')
# print("The type of this input is {}".format(type(image)))
# print("Shape: {}".format(image.shape))
# # plt.imshow(image)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # plt.imshow(image)
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # plt.imshow(gray_image, cmap='gray')
# resized_image = cv2.resize(image, (1200, 600))
# plt.imshow(resized_image)
# plt.show()

def RGB2HEX(color):
    # returns 2 digit hex number
    # print("#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2])))
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_image(image_path):
    # reads an image and returns a numpy array of length 3.
    # The first two indexes specifies pixels and the last specifies the RGB color system.
    image = cv2.imread(image_path)
    # Conversion of BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_color(image, number_of_colors):
    modified_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    # KMeans takes the input to be of two dimensions, so we use Numpy’s reshape function to reshape the image data.
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    clf = KMeans(n_clusters=number_of_colors)
    # Extractiong prediction into a variable
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    # print(counts)
    # print(sorted(counts.keys()))
    center_colors = clf.cluster_centers_
    # print(center_colors)
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    # print(ordered_colors)
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    hexa_colors_sorted = sorted(hex_colors)
    # print(hex_colors)

    plt.figure(figsize=(8, 6))
    plt.pie(counts.values(), labels=hexa_colors_sorted, colors=hexa_colors_sorted, shadow=True)
    plt.show()

#*******************************************************************************************************#

get_color(get_image('sample_image6.jpg'),8)

