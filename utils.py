import numpy as np


def points_to_image(points, class_ids, num_classes, size):
    size = np.array(size)
    points = (points * (size - 1)).round().astype(np.int32)

    image = np.zeros(((*size, num_classes)))

    for point, class_id in zip(points, class_ids):
        image[point[1], point[0], class_id] = 1

    return image


def tile_images(images):
    n = 4
    tiled = []

    for i in range(n):
        row = []
        for j in range(n):
            row.append(images[i * n + j])

        row = tf.concat(row, 1)
        tiled.append(row)

    tiled = tf.concat(tiled, 0)
    return tiled
