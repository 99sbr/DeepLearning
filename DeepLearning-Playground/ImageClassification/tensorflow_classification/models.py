import tensorflow as tf
from tensorflow import keras
import efficientnet.tfkeras as efficientnet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout


from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def __get_backbone():
    if FLAGS.model == 'DenseNet121':
        return tf.keras.applications.DenseNet121(include_top=False,
                                                 weights='imagenet',
                                                 input_shape=(FLAGS.img_width, FLAGS.img_height, 3))
    elif FLAGS.model == 'DenseNet169':
        return tf.keras.applications.DenseNet169(include_top=False,
                                                 weights='imagenet',
                                                 input_shape=(FLAGS.img_width, FLAGS.img_height, 3))
    elif FLAGS.model == 'DenseNet201':
        return tf.keras.applications.DenseNet201(include_top=False,
                                                 weights='imagenet',
                                                 input_shape=(FLAGS.img_width, FLAGS.img_height, 3))
    elif FLAGS.model == 'EfficentNetB7':
        return efficientnet.EfficientNetB7(include_top=False,
                                                    weights='noisy-student',
                                                    input_shape=(FLAGS.img_width, FLAGS.img_height, 3))


def get_model():
    backbone = __get_backbone()
    backbone.trainable = False

    model = keras.Sequential([backbone,
                              GlobalAveragePooling2D(),
                              Dense(FLAGS.no_of_classes, activation='softmax')])
    return model
