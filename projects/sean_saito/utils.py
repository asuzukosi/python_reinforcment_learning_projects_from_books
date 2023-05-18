import math, random
import numpy, cv2
# import skimage.transform
import tensorflow as tf

def cv2_resize_image(image, resized_shape=(84, 84), method='crop', crop_offset=8):
        
        height, width = image.shape
        resized_height, resized_width = resized_shape
        
        if method == 'crop':
            h = int(round(float(height) * resized_width / width))
            resized = cv2.resize(image, (resized_width, h), interpolation=cv2.INTER_LINEAR)
            crop_y_cutoff = h - crop_offset - resized_height
            cropped = resized[crop_y_cutoff : crop_y_cutoff + resized_height, :]
            return numpy.asarray(cropped, dtype=numpy.uint8)
        elif method == 'scale':
            return numpy.asarray(cv2.resize(image, (resized_width, resized_height), 
                                            interpolation=cv2.INTER_LINEAR), dtype=numpy.uint8)
        else:
            raise ValueError('Unrecognized image resize method.')
