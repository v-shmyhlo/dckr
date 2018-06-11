import tensorflow as tf
from data_loader import DataLoader
import utils
import tqdm
import itertools
import model
import argparse
import os
import numpy as np
import pickle
import losses


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--epochs', type=int, default=1000)
        self.add_argument('--experiment', type=str, required=True)


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
    ds = ds.batch(batch_size).prefetch(1)
    return ds


def map_to_image(input):
    return tf.reduce_sum(input, -1, keep_dims=True)


def build_metrics(loss, labels, logits):
    metrics = {}
    update_metrics = {}

    metrics['loss'], update_metrics['loss'] = tf.metrics.mean(loss)
    metrics['iou'], update_metrics['iou'] = tf.metrics.mean_iou(
        labels=labels, predictions=tf.to_int32(tf.nn.sigmoid(logits) > 0.5), num_classes=2)
    metrics['roc_auc'], update_metrics['roc_auc'] = tf.metrics.auc(
        labels=labels, predictions=tf.nn.sigmoid(logits), num_thresholds=10, curve='ROC')
    metrics['pr_auc'], update_metrics['pr_auc'] = tf.metrics.auc(
        labels=labels, predictions=tf.nn.sigmoid(logits), num_thresholds=10, curve='PR')

    return metrics, update_metrics


def build_summary(metrics, labels, logits, learning_rate):
    true_image = map_to_image(utils.tile_images(labels))
    pred_image = map_to_image(utils.tile_images(tf.to_float(tf.nn.sigmoid(logits) > 0.5)))

    return tf.summary.merge([
        tf.summary.scalar('learning_rate', learning_rate),
        tf.summary.scalar('loss', metrics['loss']),
        tf.summary.scalar('iou', metrics['iou']),
        tf.summary.scalar('roc_auc', metrics['roc_auc']),
        tf.summary.scalar('pr_auc', metrics['pr_auc']),
        tf.summary.image('true_image', tf.expand_dims(true_image, 0)),
        tf.summary.image('pred_image', tf.expand_dims(pred_image, 0))
    ])


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()

    training = tf.placeholder(tf.bool, [])
    global_step = tf.get_variable('global_step', initializer=0, trainable=False)
    learning_rate = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
    decrease_learning_rate = learning_rate.assign(learning_rate * 0.1)

    dl = DataLoader('./data/dataset')
    ds = build_dataset(dl, image_size=(16, 16), batch_size=16)
    train_ds = ds.skip(8).shuffle(8)
    test_ds = ds.take(8)

    iter = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
    full_init = iter.make_initializer(ds)
    train_init = iter.make_initializer(train_ds)
    test_init = iter.make_initializer(test_ds)
    input = iter.get_next()

    logits, latent = model.model(input['image'], training)

    loss = sum([
        # TODO: cleanup
        # tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=input['image'], logits=logits)),
        # tf.reduce_mean(jaccard_loss(labels=input['image'], logits=logits)),
        # tf.reduce_mean(focal_sigmoid_cross_entropy_with_logits(labels=input['image'], logits=logits)),
        tf.reduce_mean(losses.balanced_sigmoid_cross_entropy_with_logits(labels=input['image'], logits=logits)),
        tf.reduce_mean(losses.dice_loss(labels=input['image'], logits=logits)),

        tf.losses.get_regularization_loss()
    ])

    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss, global_step)

    metrics, update_metrics = build_metrics(loss, labels=input['image'], logits=logits)
    summary = build_summary(metrics, input['image'], logits, learning_rate)
    locals_init = tf.local_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess, tf.summary.FileWriter(
            os.path.join('./tf_log', args.experiment, 'train')) as train_writer, tf.summary.FileWriter(
        os.path.join('./tf_log', args.experiment, 'test')) as test_writer:
        def iterate(desc, init, writer, iteration_step, feed_dict):
            sess.run(init)
            sess.run(locals_init)

            for _ in tqdm.tqdm(itertools.count(), desc=desc):
                try:
                    _, step = sess.run([iteration_step, global_step], feed_dict)
                except tf.errors.OutOfRangeError:
                    break

            sess.run(init)

            m, s = sess.run([metrics, summary], {training: False})
            writer.add_summary(s, step)
            writer.flush()

            print()
            print('({}) step: {}, iou: {:.4f}, loss: {:.4f}'.format(desc, step, m['iou'], m['loss']))

            return m


        restore_path = tf.train.latest_checkpoint(os.path.join('./tf_log', args.experiment))
        if restore_path:
            saver.restore(sess, restore_path)
            print('model restored from {}'.format(restore_path))
        else:
            sess.run(tf.global_variables_initializer())

        m_prev = []
        for epoch in range(args.epochs):
            # training
            iterate(
                desc='train',
                init=train_init,
                writer=train_writer,
                iteration_step=[train_step, update_metrics],
                feed_dict={training: True})

            # testing
            m = iterate(
                desc='test',
                init=test_init,
                writer=test_writer,
                iteration_step=update_metrics,
                feed_dict={training: False})

            # save model
            saver.save(sess, os.path.join('./tf_log', args.experiment, 'model.ckpt'))
            # stop training when no improvement
            m_prev.append(m)
            if len(m_prev) > 1:
                iou = np.array([m['iou'] for m in m_prev[-5:]])
                if iou.max() - iou.min() < 0.0005:
                    print('stopped at iou delta {:.4f}'.format(iou.max() - iou.min()))
                    break

        # save latent vectors for clustering
        sess.run(full_init)
        data = {
            'files': [],
            'images': [],
            'vectors': []
        }
        for _ in tqdm.tqdm(itertools.count(), desc='computing vectors'):
            try:
                f, i, l = sess.run([input['file'], input['image'], latent], {training: False})
                data['files'].append(f)
                data['images'].append(i)
                data['vectors'].append(l)
            except tf.errors.OutOfRangeError:
                break

        data = {k: np.concatenate(data[k], 0) for k in data}
        assert data['files'].shape[0] == data['images'].shape[0] == data['vectors'].shape[0] == len(dl)

        with open('./data/output.pickle', 'wb') as f:
            pickle.dump(data, f)
