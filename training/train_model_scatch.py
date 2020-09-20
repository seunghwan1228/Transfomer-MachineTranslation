import tensorflow as tf
import tqdm
import time

from models.transformer import TransformerModel
from models.masking import create_padding_mask, create_combined_mask
from utils.load_config import LoadConfig
from training.training_utils import CustomSchedule
from prepare_data.create_data import CreateData





# Config dict and model is for reference
config_dict = LoadConfig('conf').load_config()


# Load Data
dataset_name = config_dict['dataset_name']
data_creator = CreateData(config_path='conf')
train_datasets, valid_datasets, test_datasets = data_creator.create_all()


# Define Model
model = TransformerModel(encoder_vocab_size=data_creator.tokenizer.lang_one_vocab_size,
                         decoder_vocab_size=data_creator.tokenizer.lang_two_vocab_size,
                         encoder_max_pos=config_dict['max_pos_length'],
                         decoder_max_pos=config_dict['max_pos_length'],
                         num_heads=config_dict['num_heads'],
                         model_dim=config_dict['model_dim'],
                         feed_forward_dim=config_dict['feed_forward_dim'],
                         dropout_rate=config_dict['dropout_rate'],
                         mha_concat_query=config_dict['mha_concat_query'],
                         n_layers=config_dict['n_layers'],
                         debug=config_dict['debug'])

# Learning Rate Schedule
model_learning_rate = CustomSchedule(config_dict['model_dim'])
model_optimizer = tf.keras.optimizers.Adam(learning_rate=model_learning_rate,
                                           beta_1=0.9,
                                           beta_2=0.98,
                                           epsilon=1e-9)


# Loss Object
#  If reduction is NONE, this has shape [batch_size, d0, .. dN-1];
#  otherwise, it is scalar.
#  (Note dN-1, because all loss function reduce by1 dimension, usually axis=-1)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')

def loss_function(y_true, y_pred):
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
    mask = tf.cast(tf.math.logical_not(tf.math.equal(y_true, 0)), dtype=tf.int64)
    loss_ = loss_object(y_true, y_pred, sample_weight=mask)
    return loss_



# Training Metrics
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='Train Accuracy')
train_loss = tf.keras.metrics.Mean(name='Train Loss')


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.
train_input_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                         tf.TensorSpec(shape=(None, None), dtype=tf.int64)]


# Train Step - My model includes python syntax, so could not compile as tensorflow graph..
def model_train_step(encoder_input_seq, decoder_target_seq):
    decoder_input = decoder_target_seq[:, :-1]  # <sos> 1, 2, 3, 4, 5
    decoder_target = decoder_target_seq[:, 1:]  # 1, 2, 3, 4, <eos>

    encoder_padding_mask = create_padding_mask(encoder_input_seq)
    decoder_padding_one_mask = create_combined_mask(decoder_input)

    with tf.GradientTape() as tape:
        prediction, _, _ = model(encoder_input = encoder_input_seq,
                                 decoder_input = decoder_input,
                                 encoder_mask = encoder_padding_mask,
                                 decoder_mask_one = decoder_padding_one_mask,
                                 decoder_mask_two = encoder_padding_mask,
                                 drop_n_heads = config_dict['drop_n_heads'],
                                 training=True)

        loss_value = loss_function(decoder_target, prediction)

    model_gradient = tape.gradient(loss_value, model.trainable_variables)
    model_optimizer.apply_gradients(zip(model_gradient, model.trainable_variables))
    train_accuracy(decoder_target, prediction)
    train_loss(loss_value)


# Checkpoint Related
checkpoint = tf.train.Checkpoint(model=model,
                                 model_optimizer=model_optimizer)

model_step = tf.Variable(0, name='model_step')
checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                directory=config_dict['log_dir'],
                                                max_to_keep=config_dict['max_to_keep'],
                                                keep_checkpoint_every_n_hours=config_dict['keep_n_hours'],
                                                step_counter=model_step)



def train_start():
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print('Restored Latest Checkpoint')
    else:
        print('Initializing Training from Scratch')
    # Perform Training
    for e in tqdm.tqdm(range(config_dict['epoch'])):
        START = time.time()

        train_accuracy.reset_states()
        train_loss.reset_states()

        for (batch, (encoder_input_seq, decoder_target_seq)) in enumerate(train_datasets):
            model_train_step(encoder_input_seq=encoder_input_seq,
                             decoder_target_seq=decoder_target_seq)

            if batch % 50 == 0:
                print(f'Epoch: {e}\tLoss: {train_loss.result()}\tAccuracy: {train_accuracy.result()}')

        if (e+1) % 5 == 0 :
            checkpoint_manager.save()
        print(f'Epoch: {e}\tLoss: {train_loss.result()}\tAccuracy: {train_accuracy.result()}')
        print(f'Duration: {time.time() - START}')



train_start()