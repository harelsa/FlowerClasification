from tkinter.ttk import Combobox

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from tkinter import filedialog
from tkinter import *


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

# model_path = r'‪C:\Users\harel_000\Desktop\flowers_model.h5'
# flower_path = r'C:\Users\harel_000\PycharmProjects\FlowerClasification\flowers\flowers'
#
# predict(model_path, flower_path)

class MyWindow:
    def __init__(self, win):
        self.lbl1=Label(win, text='Set Module Path :')
        self.lbl2=Label(win, text='Set Flower Path :')
        self.lbl3=Label(win, text='Result:')
        self.module_path=Text(win,width=40, height=1)
        self.flower_path=Text(win,width=40, height=1)
        self.result=Text(win,width=30, height=10)
        self.t3=Entry()
        self.btn1 = Button(win, text='module_btn')
        self.btn2=Button(win, text='flower_btn')
        self.btn3 = Button(win, text='predict_btn')
        self.lbl1.place(x=30, y=50)
        self.module_path.place(x=150, y=50)
        self.lbl2.place(x=30, y=100)
        self.flower_path.place(x=150, y=100)
        self.result.place(x=200, y=200)
        self.b1=Button(win, text='...', command=self.add_module)
        self.b2=Button(win, text='...', command=self.add_flowers)
        self.b3=Button(win, text='predict', command=self.predict)
        # self.b2.bind('<Button-1>', self.sub)
        self.b1.place(x=500, y=45)
        self.b2.place(x=500, y=95)
        self.b3.place(x=450, y=145)
        self.lbl3.place(x=100, y=200)
        self.t3.place(x=200, y=200)
    def add_module(self):
        # self.module_path.insert(END,filedialog.askopenfilename(initialdir="/", title="Select file",
        #                             filetypes=(("data files", "*.h5"), ("all files", "*.*"))))
        self.module_path.insert(END , r'C:/Users/harel_000/Desktop/flowers_model.h5')
    def add_flowers(self):
        # self.flower_path.insert(END,filedialog.askdirectory())
        self.flower_path.insert(END, r'C:/Users/harel_000/Desktop/flowers/flowers')
    def predict(self):
        model_path_str = self.module_path.get(1.0,END)
        flower_path_str = self.flower_path.get(1.0,END)
        predict(model_path_str, flower_path_str)


window=Tk()
mywin=MyWindow(window)
window.title('Flower Classification')
window.geometry("600x500+10+10")

window.mainloop()

