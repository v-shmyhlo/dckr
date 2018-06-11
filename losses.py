def dice_loss(labels, logits, smooth=1, name='dice_loss'):
    with tf.name_scope(name):
        probs = tf.nn.sigmoid(logits)

        intersection = tf.reduce_sum(labels * probs, [1, 2, 3])
        union = tf.reduce_sum(labels, [1, 2, 3]) + tf.reduce_sum(probs, [1, 2, 3])

        coef = (2 * intersection + smooth) / (union + smooth)
        loss = 1 - coef

        return loss


def focal_sigmoid_cross_entropy_with_logits(labels, logits, focus=2.0, alpha=0.25,
                                            name='focal_sigmoid_cross_entropy_with_logits'):
    with tf.name_scope(name):
        alpha = tf.ones_like(labels) * alpha
        labels_eq_1 = tf.equal(labels, 1)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        prob = tf.nn.sigmoid(logits)
        a_balance = tf.where(labels_eq_1, alpha, 1 - alpha)
        prob_true = tf.where(labels_eq_1, prob, 1 - prob)
        modulating_factor = (1.0 - prob_true)**focus

        return a_balance * modulating_factor * loss


def balanced_sigmoid_cross_entropy_with_logits(labels, logits, name='balanced_sigmoid_cross_entropy_with_logits'):
    with tf.name_scope(name):
        num_positive = tf.reduce_sum(tf.to_float(tf.equal(labels, 1)))
        num_negative = tf.reduce_sum(tf.to_float(tf.equal(labels, 0)))

        weight_positive = num_negative / (num_positive + num_negative)
        weight_negative = num_positive / (num_positive + num_negative)
        ones = tf.ones_like(logits)
        weight = tf.where(tf.equal(labels, 1), ones * weight_positive, ones * weight_negative)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = loss * weight

        return loss
