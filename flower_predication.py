from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras_preprocessing.image import ImageDataGenerator


def predict(model_path, flower_path):
    model = load_model(model_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    train = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest',
        validation_split=0
    )
    img_size = 128
    batch_size = 1
    pred_gen = train.flow_from_directory(flower_path, target_size=(img_size, img_size), batch_size=batch_size,
                                          class_mode='categorical')

    i = 3000  ### Full run need to change to pred.gen.n
    while i < 3010:
        img = image.load_img(pred_gen.filepaths[i], target_size=(img_size, img_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x)
        print(pred_gen.filenames[i])
        print(pred)
        print(pred.argmax())
        # 0 - daisy
        # 1 - dandelion
        # 2 - rose
        # 3 - sunflower
        # 4 - tulip
        i += 1


### Main ###

model_path = r'C:\Users\eladfo\PycharmProjects\untitled\flowers_model.h5'
flower_path = r'C:\Users\eladfo\PycharmProjects\untitled\flowers'

predict(model_path, flower_path)