import numpy as np

#def convert_image_to_greyscale(cifar_image):

def convert_set_to_greyscale(cifar_set):
    converted_set = np.empty((cifar_set.shape[0], 1, cifar_set.shape[2], cifar_set.shape[3]), 'uint8')
    for image_index, image in enumerate(cifar_set):
        for row_index, row in enumerate(image[0]):
            for pixel_index, pixel in enumerate(row):
                grey = (0.2126 * cifar_set[image_index, 0, row_index, pixel_index]) +\
                       (0.7152 * cifar_set[image_index, 1, row_index, pixel_index]) +\
                       (0.0722 * cifar_set[image_index, 2, row_index, pixel_index])
                converted_set[image_index, 0, row_index, pixel_index] = int(round(grey))
    print 'Converted ', len(converted_set), ' images to greyscale.'
    return converted_set
