import csv
import numpy as np
import os

classes = ['1', '13', '17', '19', '2', '23', '24--4111', '24-15', '24-4', '24-5', '24-51', '24-52', '24-53', '24-57',
           '24-58', '24-59', '25', '5', '9']

min_point = -2268850
max_point = 9655840


class DataLoader(object):
    def __init__(self, path):
        self._path = path

    def __iter__(self):
        files = self._get_files()

        for file in files:
            with open(os.path.join(self._path, file))as f:
                points = [(int(x), int(y), classes.index(z)) for x, y, z in csv.reader(f)]

            points = np.array(points)
            points, class_ids = points[..., :2], points[..., 2]
            points -= min_point
            points = points / (max_point - min_point)

            yield {
                'file': file,
                'points': points,
                'class_ids': class_ids
            }

    def __len__(self):
        return len(self._get_files())

    @property
    def num_classes(self):
        return len(classes)

    def _get_files(self):
        return os.listdir(self._path)


if __name__ == '__main__':
    dl = DataLoader('./data/dataset')

    all_points = []
    all_class_ids = []

    for input in dl:
        all_points.append(input['points'])
        all_class_ids.append(input['class_ids'])

    all_points = np.concatenate(all_points, 0)
    all_class_ids = np.concatenate(all_class_ids, 0)
    print(all_points.shape, all_points.dtype)
    print(all_class_ids.shape, all_class_ids.dtype)
    print(all_points.min(), all_points.max())
