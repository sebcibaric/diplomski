import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import json
import sys
from glob import glob

gpus = tf.config.list_physical_devices('GPU')
print('gpus = {}'.format(gpus[0]))
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=768)])

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
        sys.exit(0)

dataset_train = './dataset/train'
dataset_valid = './dataset/valid'

IMAGE_SIZE = 224
BATCH_SIZE = 20
EPOCH_SIZE = 10

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255
)

data_valid_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255
)

train_generator = data_generator.flow_from_directory(
    dataset_train,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
)

valid_generator = data_valid_generator.flow_from_directory(
    dataset_valid,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
)

print(train_generator.class_indices)

with open('data.json', 'w') as outputfile:
    json.dump(train_generator.class_indices, outputfile, indent=4)

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

mobilenet = tf.keras.applications.MobileNetV2(weights='imagenet',
                                              input_shape=IMG_SHAPE,
                                              include_top=False)

classes = glob("dataset/train/*")

# base_model = VGG16(weights='imagenet', input_shape=IMG_SHAPE, include_top=False)
# mobilenet = VGG16(weights='imagenet', input_shape=IMG_SHAPE, include_top=False)

mobilenet.trainable = False
# base_model.trainable = False

# for layer in base_model.layers:
#     layer.trainable = False
#
# x = layers.Flatten()(base_model.output)
#
# # Add a fully connected layer with 512 hidden units and ReLU activation
# x = layers.Dense(512, activation='relu')(x)
#
# # Add a dropout rate of 0.5
# x = layers.Dropout(0.5)(x)
#
# # Add a final sigmoid layer with 1 node for classification output
# x = layers.Dense(len(classes), activation='relu')(x)
#
# model = Model(base_model.input, x)
#
# # model.compile(optimizer=RMSprop(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model = tf.keras.Sequential([
    mobilenet,
    # tf.keras.layers.Conv2D(32, 3, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(train_generator, epochs=EPOCH_SIZE, validation_data=valid_generator)

model.save('./model.h5')
