import json
from glob import glob
from tensorflow import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.core import Dropout

gpus = config.list_physical_devices('GPU')
if gpus:
    print('gpus = {}'.format(gpus[0]))
    try:
        config.set_logical_device_configuration(
            gpus[0],
            [config.LogicalDeviceConfiguration(memory_limit=512)])

        logical_gpus = config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

dataset_train = './dataset/train'
dataset_valid = './dataset/valid'

IMAGE_SIZE = 224
BATCH_SIZE = 20
EPOCH_SIZE = 10

data_generator = ImageDataGenerator(
    rescale=1. / 255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True,
)

data_valid_generator = ImageDataGenerator(
    rescale=1. / 255
)

train_generator = data_generator.flow_from_directory(
    dataset_train,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = data_valid_generator.flow_from_directory(
    dataset_valid,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print(train_generator.class_indices)

with open('data.json', 'w') as outputfile:
    json.dump(train_generator.class_indices, outputfile, indent=4)

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

classes = glob("dataset/train/*")

model = Sequential()
model.add(Conv2D(4, (3, 3), activation='relu', input_shape=IMG_SHAPE, padding="same"))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2, 2), 2))
model.add(Conv2D(8, (3, 3), activation='relu', padding="same"))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2, 2), 2))
model.add(Flatten())
model.add(Dense(len(classes), activation='softmax'))

callbacks = [EarlyStopping(monitor='val_accuracy', patience=3)]

model.compile(optimizer=Adam(learning_rate=0.00001, decay=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(train_generator,
    epochs=EPOCH_SIZE,
    validation_data=valid_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(valid_generator),
    callbacks=callbacks
)

model.save('model.h5')
