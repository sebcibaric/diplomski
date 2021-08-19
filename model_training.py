import tensorflow as tf
import json
from glob import glob

dataset_train = 'dataset/train/'
dataset_valid = 'dataset/valid/'

IMAGE_SIZE = 224
BATCH_SIZE = 20
EPOCH_SIZE = 10

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1. / 255,
    validation_split = 0.2
)

train_generator = data_generator.flow_from_directory(
    dataset_train,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size = BATCH_SIZE,
    subset = 'training'
)

valid_generator = data_generator.flow_from_directory(
    dataset_valid,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size = BATCH_SIZE,
    subset = 'validation'
)

print(train_generator.class_indices)

with open('data.json', 'w') as outputfile:
    json.dump(train_generator.class_indices, outputfile, indent=4)

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

mobilenet = tf.keras.applications.MobileNetV2(weights = 'imagenet',
                                          input_shape = IMG_SHAPE,
                                          include_top = False) 

mobilenet.trainable = False

classes = glob("dataset/train/*")

model = tf.keras.Sequential([
    mobilenet,
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

model.summary()

model.fit(train_generator, epochs = EPOCH_SIZE, validation_data = valid_generator)

# mobilenet.trainable = True

# fine_tune_at = 100

# for layer in mobilenet.layers[:fine_tune_at]:
#     layer.trainable = False

# model.compile(optimizer = tf.keras.optimizers.Adam(1e-5),
#               loss = 'categorical_crossentropy',
#               metrics = ['accuracy'])

# model.summary()

# model.fit(train_generator, epochs = EPOCH_SIZE // 2, validation_data = valid_generator)

model.save('./model.h5')
