import numpy as np


def points_to_image(points, class_ids, num_classes, size):
    size = np.array(size)
    points = (points * (size - 1)).round().astype(np.int32)
    image = np.zeros(((*size, num_classes)))

    for point, class_id in zip(points, class_ids):
        image[point[1], point[0], class_id] = 1

    return image

# def one_hot(indices, num_classes):
#     vectors = np.eye(num_classes)
#     return vectors[indices]
