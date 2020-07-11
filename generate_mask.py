from random import randint
import cv2
import numpy as np
import matplotlib.pyplot as plt


''' 
    Code for generating random mask is borrowed from 
    https://github.com/MathiasGruber/PConv-Keras
    You can change the parameter for desire mask area(Now is about 20%)
'''
def random_mask(height, width, channels=3):
    """Generates a random irregular mask with lines, circles and elipses"""
    img = np.zeros((height, width, channels), np.uint8)

    # Draw random lines
    for _ in range(randint(7, 9)):
        x1, x2 = randint(1, width), randint(1, width)
        y1, y2 = randint(1, height), randint(1, height)
        thickness = randint(7, 10)
        cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

    # Draw random circles
    for _ in range(randint(7, 9)):
        x1, y1 = randint(1, width), randint(1, height)
        radius = randint(8, 12)
        cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

    # Draw random ellipses
    for _ in range(randint(7, 9)):
        x1, y1 = randint(1, width), randint(1, height)
        s1, s2 = randint(1, width), randint(1, height)
        a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
        thickness = randint(8, 12)
        cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)
    return 1 - img


if __name__ == '__main__':
    dir = './random_mask/mask_slim/'
    size = 512
    for i in range(5000):
        sample_mask = random_mask(size, size)
        sample_mask *= 255
        cv2.imwrite(dir + "mask_{}.jpg".format(str(i)), sample_mask)
        read = cv2.imread(dir + "mask_{}.jpg".format(str(i)))
        # cv2.imshow('window', read)
        # cv2.imshow('window2', sample_mask)
        # cv2.waitKey(0)

        # sample_mask = cv2.cvtColor(sample_mask, cv2.COLOR_RGB2GRAY)
        # valid_area = np.sum(sample_mask)
        # percentage = valid_area / (size * size)
        # print(1 - percentage)
        # plt.imshow(sample_mask * 255)
        # plt.show()
    print("Created.")
