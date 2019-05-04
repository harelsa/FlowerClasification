from tkinter.ttk import Combobox
import csv
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from tkinter import filedialog
from tkinter import *



class MyWindow:
    def __init__(self, win):
        self.lbl1=Label(win, text='Set Module Path :')
        self.lbl2=Label(win, text='Set Flower Path :')
        self.lbl3=Label(win, text='Results: ( Roll down the mouse to go through the results) ')
        self.lbl4=Label(win, text='Save Csv to:')
        self.module_path=Text(win,width=40, height=1)
        self.flower_path=Text(win,width=40, height=1)
        self.csv_path=Text(win,width=40, height=1)
        self.result=Text(win,width=80, height=20)
        # self.t3=Entry()
        self.btn1 = Button(win, text='module_btn')
        self.btn2=Button(win, text='flower_btn')
        self.btn3 = Button(win, text='predict_btn')
        self.btn4 = Button(win, text='csv_btn')
        self.lbl1.place(x=30, y=50)
        self.lbl4.place(x=30, y=150)
        self.module_path.place(x=150, y=50)
        self.lbl2.place(x=30, y=100)
        self.flower_path.place(x=150, y=100)
        self.csv_path.place(x=150, y=150)
        self.result.place(x=20, y=200)
        self.b1=Button(win, text='...', command=self.add_module)
        self.b2=Button(win, text='...', command=self.add_flowers)
        self.b4=Button(win, text='...', command=self.add_csv)
        self.b3=Button(win, text='predict', command=self.predict)
        # self.b2.bind('<Button-1>', self.sub)
        self.b1.place(x=500, y=45)
        self.b2.place(x=500, y=95)
        self.b3.place(x=600, y=145)
        self.b4.place(x=500, y=145)
        self.lbl3.place(x=25, y=180)
        self.csv_path.insert(END, 'set by default to project dir')
        # self.t3.place(x=200, y=200)
    def add_module(self):
        self.module_path.delete('1.0', END)
        self.module_path.insert(END,filedialog.askopenfilename(initialdir="/", title="Select file",
                                                filetypes=(("data files", "*.h5"), ("all files", "*.*"))))
        # self.module_path.insert(END , r"C:/Users/harel_000/Desktop/flowers_model.h5")
    def add_flowers(self):
        self.flower_path.delete('1.0', END)
        self.flower_path.insert(END,filedialog.askdirectory())
        # self.flower_path.insert(END, r"C:/Users/harel_000/Desktop/flowers/flowers")
    def add_csv(self):
        self.csv_path.delete('1.0', END)
        self.csv_path.insert(END,filedialog.askdirectory())

    def predict(self):
        model_path_str = self.module_path.get(1.0,END)
        model_path_str = model_path_str.replace('\n', '')
        flower_path_str = self.flower_path.get(1.0,END)
        flower_path_str = flower_path_str.replace('\n', '')
        predict(model_path_str, flower_path_str)


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

    i = 0  ### Full run need to change to pred.gen.n
    mywin.result.delete('1.0', END)
    csv_path = mywin.csv_path.get(1.0,END)
    csv_path = csv_path.replace('\n', '')
    if csv_path == 'set by default to project dir':
        csv_path = ''
    else :
        csv_path = csv_path + '//'
    with open(csv_path + 'predict.csv', 'w', newline='') as csvfile:
        wr = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        while i < pred_gen.n:
            img = image.load_img(pred_gen.filepaths[i], target_size=(img_size, img_size))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            pred = model.predict(x)
            print(pred_gen.filenames[i])
            mywin.result.insert(END,pred_gen.filenames[i] + ",")
            # print(pred)
            print(pred.argmax())
            mywin.result.insert(END, flower_dic[pred.argmax()] + "\n")
            wr.writerow([pred_gen.filenames[i], flower_dic[pred.argmax()]])
            # 0 - daisy
            # 1 - dandelion
            # 2 - rose
            # 3 - sunflower
            # 4 - tulip
            i += 1


### Main ###

# model_path = "C:\\Users\\harel_000\\Desktop\\flowers_model.h5"
# flower_path = "C:\\Users\\harel_000\\PycharmProjects\\FlowerClasification\\flowers\\flowers"
flower_dic = [ "daisy",
               "dandelion",
               "rose",
               "sunflower",
               "tulip"]
# predict(model_path, flower_path)





window=Tk()
mywin=MyWindow(window)
window.title('Flower Classification')
window.geometry("700x600+10+10")

window.mainloop()

