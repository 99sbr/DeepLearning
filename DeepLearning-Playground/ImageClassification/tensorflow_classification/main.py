import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.platform import flags

from tensorflow_classification.dataloader import get_train_and_valid_data
from tensorflow_classification.models import get_model

AUTO = tf.data.experimental.AUTOTUNE

FLAGS = flags.FLAGS

# Dataset Specifications:
flags.DEFINE_string('img_dir', '/Users/subir/Codes/DeepLearning/DeepLearning-Playground/ImageClassification/Snakes in '
                               'Hood/dataset/train/',
                    'Directory where all the training images are irrespective of classes')
flags.DEFINE_string('dataframe', '/Users/subir/Codes/DeepLearning/DeepLearning-Playground/ImageClassification/Snakes '
                                 'in Hood/dataset/train.csv',
                    'CSV file containing image names and filenames')

flags.DEFINE_integer('img_width', 299, 'Training Image Width')
flags.DEFINE_integer('img_height', 299, 'Training Image Height')
flags.DEFINE_integer('batch_size', 32, 'Batch Size for Minibatches')
flags.DEFINE_float('validation_split', 0.1, 'Train-Test Validation Split')
flags.DEFINE_bool('stratify', True, 'Stratify training set')

# Model Specifications:
flags.DEFINE_string('model', 'EfficentNetB7', 'Backbone Feature Training')
flags.DEFINE_integer('no_of_classes', 35, 'Number of Output Classes')
flags.DEFINE_string('optimizer', 'adam', 'Optimizer in training routine')

# Runtime Specifications:
flags.DEFINE_integer('epochs', 2, 'Number of Epochs to train')
flags.DEFINE_integer('verbose', 1, 'Verbosity at Training Time')
flags.DEFINE_bool('model_summary', True, 'Print Model Summary')
flags.DEFINE_bool('finetune', False, 'True: Transfer learning, False: ')

# Tensorboard Parameters
flags.DEFINE_bool('enable_log', True, 'Whether to log the training process')
flags.DEFINE_string('logdir', './log_directory/',
                    'directory to store Tensorboard log')

# Saving and Reloading Parameters
flags.DEFINE_bool('resume_training', False,
                  'Whether to load previously trained weights to train further')

if FLAGS.enable_log:
    print('Adding Tensorborad')
    callbacks = [TensorBoard(log_dir=FLAGS.logdir, histogram_freq=1)]

# Loading Training Data
train_data, valid_data = get_train_and_valid_data()

if FLAGS.resume_training:
    model = tf.keras.models.load_model(f'saved_model/{FLAGS.model}')
    print(
        f'Pretrained weights are loaded successfully from saved_model/{FLAGS.model}')
else:
    # Get the model architecture
    model = get_model()
    model.compile(optimizer=FLAGS.optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

if FLAGS.model_summary:
    model.summary()

if FLAGS.finetune:
    model.get_layer(index=0).trainable = True
    if FLAGS.no_of_classes >3:
        model.compile(optimizer=FLAGS.optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=FLAGS.optimizer,
                      loss='binary_corssentropy', metrics=['accuracy'])
    print('\n\nBackbone is trainable. Backbone Finetuning!!!!\n\n')
else:
    model.get_layer(index=0).trainable = False
    if FLAGS.no_of_classes > 3:
        model.compile(optimizer=FLAGS.optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=FLAGS.optimizer,
                      loss='binary_corssentropy', metrics=['accuracy'])
    print('\n\nBackbone is frozen. Transfer learning!!!!\n\n')

history = model.fit(train_data,
                    validation_data=valid_data,
                    steps_per_epoch=len(train_data),
                    validation_steps=len(valid_data),
                    epochs=FLAGS.epochs,
                    verbose=FLAGS.verbose,
                    callbacks=callbacks
                    )

model.save(f'saved_model/{FLAGS.model}')
print('Success')
