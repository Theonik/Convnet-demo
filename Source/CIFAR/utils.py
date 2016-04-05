import numpy as np
from math import sqrt
#def convert_image_to_greyscale(cifar_image):

def convert_set_to_greyscale(cifar_set, method=0, gamma=1.0):
    converted_set = np.empty((cifar_set.shape[0], 1, cifar_set.shape[2], cifar_set.shape[3]), 'float32')
    for image_index, image in enumerate(cifar_set):
        for row_index, row in enumerate(image[0]):
            for pixel_index, pixel in enumerate(row):
                grey = 0.0
                if method == 0:  # rec.709 luminance
                    grey = (0.2126 * cifar_set[image_index, 0, row_index, pixel_index]) + \
                           (0.7152 * cifar_set[image_index, 1, row_index, pixel_index]) + \
                           (0.0722 * cifar_set[image_index, 2, row_index, pixel_index])
                elif method == 1:  # NTSC/W3C luminance
                    grey = (0.299 * cifar_set[image_index, 0, row_index, pixel_index]) + \
                           (0.587 * cifar_set[image_index, 1, row_index, pixel_index]) + \
                           (0.114 * cifar_set[image_index, 2, row_index, pixel_index])
                elif method == 2:
                    grey = sqrt(((0.299 * cifar_set[image_index, 0, row_index, pixel_index]) ** 2) +
                                ((0.587 * cifar_set[image_index, 1, row_index, pixel_index]) ** 2) +
                                ((0.114 * cifar_set[image_index, 2, row_index, pixel_index]) ** 2))
                elif method == 3:
                    grey = sqrt(((0.2126 * cifar_set[image_index, 0, row_index, pixel_index]) ** 2) +
                                ((0.7152 * cifar_set[image_index, 1, row_index, pixel_index]) ** 2) +
                                ((0.0722 * cifar_set[image_index, 2, row_index, pixel_index]) ** 2))
                elif method == 4:  # Simple mean of RGB
                    grey = ((cifar_set[image_index, 0, row_index, pixel_index]) +
                            (cifar_set[image_index, 1, row_index, pixel_index]) +
                            (cifar_set[image_index, 2, row_index, pixel_index])) / 3
                else:
                    print 'Error: This is not a valid conversion mode.\n Reverting to colour.'
                    return cifar_set.astype('float32') / 255

                converted_set[image_index, 0, row_index, pixel_index] = np.float32(grey/255)
    print 'Converted ', len(converted_set), ' images to greyscale.'
    return converted_set
