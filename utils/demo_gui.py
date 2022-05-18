from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os

root = Tk()
root.geometry("1350x480")
root.resizable(width=True, height=True)
root.title('Registration Demo')

_canvas = Canvas(root,height=200,width=200)
_canvas.pack
_canvas.create_rectangle(30 ,  70 ,  190 ,  170,  outline="black", stipple='gray25')
def open_img1():
    image=Image.open('s.PNG')
    img=image.resize((400, 400))
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.place(x=5, y=50)

def open_img2():
    image=Image.open('t.PNG')
    img=image.resize((400, 400))
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.place(x=460, y=50)

def open_img3():
    image=Image.open('r.PNG')
    img=image.resize((400, 400))
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.place(x=915, y=50)

btn1 = Button(root, text='Capture Image 1', command=open_img1)
btn1.place(x=120, y=10)

btn2 = Button(root, text='Capture Image 2', command=open_img2)
btn2.place(x=600, y=10)


btn3 = Button(root, text='Estimate Registration', command=open_img3)
btn3.place(x=1050, y=10)

root.mainloop()