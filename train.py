import tensorflow as tf
from data_loader import DataLoader
import utils
import tqdm
import itertools
import model
import argparse


def build_dataset(dl, image_size, batch_size):
    def gen():
        for input in dl:
            yield {
                'file': input['file'].encode('utf-8'),
                'image': utils.points_to_image(input['points'], input['class_ids'], dl.num_classes, image_size)

            }

    ds = tf.data.Dataset.from_generator(
        gen,
        output_types={'file': tf.string, 'image': tf.float32},
        output_shapes={'file': [], 'image': [*image_size, dl.num_classes]})
    ds = ds.prefetch(1).batch(batch_size)
    return ds


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--epochs', type=int, default=100)


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()

    training = tf.placeholder(tf.bool, [])
    global_step = tf.get_variable('global_step', 0, trainable=False)

    dl = DataLoader('./data/dataset')
    ds = build_dataset(dl, image_size=(24, 24), batch_size=32)

    iter = ds.make_initializable_iterator()
    input = iter.get_next()

    # metrics, update_metrics = build_metrics(input)
    logits = model.model(input['image'])

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=input['image'], logits=logits)

    optimizer = tf.train.AdamOptimizer()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss)

    summary = tf.summary.merge([
        tf.summary.scalar('loss', loss),
        tf.summary.image('true_image', tf.reduce_sum(input['image'], -1, keep_dims=True)),
        tf.summary.image('pred_image', tf.reduce_sum(tf.to_float(tf.nn.sigmoid(logits) > 0.5), -1, keep_dims=True))
    ])

    with tf.Session() as sess, tf.summary.FileWriter('./tf_log') as writer:
        for epoch in range(args.epochs):
            sess.run(iter.initializer)

            for _ in tqdm.tqdm(itertools.count()):
                try:
                    _, step = sess.run([train_step, global_step], {training: True})
                except tf.errors.OutOfRangeError:
                    break

            s = sess.run(summary, {training: True})
            writer.add_summary(s, step)
