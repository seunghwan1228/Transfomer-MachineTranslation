import tensorflow as tf



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    '''
    Attention is all you needs
    https://arxiv.org/abs/1706.03762
    5.3 Optimizer - learning rate
    '''
    def __init__(self, model_dim, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.model_dim = tf.cast(model_dim, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.model_dim) * tf.math.minimum(arg1, arg2)



def sparse_crossetropy_loss(y_true, y_pred):
    '''
    Optional sample_weight acts as a coefficient for the loss.
    If a scalar is provided, then the loss is simply scaled by the given value.
    If sample_weight is a tensor of size [batch_size],
    then the total loss for each sample of the batch is rescaled
    by the corresponding element in the sample_weight vector.
     If the shape of sample_weight is [batch_size, d0, .. dN-1]
     (or can be broadcasted to this shape),
    then each loss element of y_pred is scaled by the corresponding value of sample_weight.
    '''
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction='none')
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = loss_object(y_true, y_pred, sample_weight=mask)
    return loss_


def sparse_accuracy_metrics(y_true, y_pred, name):
    acc_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name=name)
    return acc_metrics(y_true, y_pred)


def loss_metrics(loss_value):
    mean_metrics = tf.keras.metrics.Mean()
    return mean_metrics(loss_value)


