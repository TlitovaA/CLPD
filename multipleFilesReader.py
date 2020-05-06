from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import tkinter
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
from tkinter import messagebox

root = tkinter.Tk()
root.geometry('200x200') 

files=[]

def choose_else():
    file = askopenfile(mode ='r', initialdir =  "/diplom/", title = "Select A File",
                       filetypes =[('TEXT', '*.txt'),("all files","*.*")])
    label = Label(text = file.name)
    label.pack()
        
    if file is not None: 
        content = file.read() 
        print(content)
        files.append(file)
    else:
        print("No file selected")
 #   messagebox.showinfo( "Selection", "Choose the 2nd file for analysis")

  
def open_file(): 
    file = askopenfile(mode ='r', initialdir =  "/", title = "Select A File",
                       filetypes =[('TEXT', '*.txt'),("all files","*.*")])

    label = Label(text = file.name)
    label.pack()

    if file is not None: 
        content = file.read() 
        print(content)
        files.append(file)
    else:
        print("No file selected")

def close():
    global root
    root.quit()

open1 = Button(root, text ="Open the 1st file", command = choose_else)  
open2 = Button(root, text ='Open the 2nd file', command = open_file)
ok = Button(root, text ='Ok', command = close)

open1.pack(side = TOP, pady = 10) 
open2.pack(side = TOP, pady = 10) 
ok.pack(side = TOP, pady = 10) 

root.mainloop()
root.iconify()

file1 = open(files[0].name, encoding='utf-8')
file2 = open(files[1].name, encoding='utf-8')
print(file1.read())
print(file2.read()) 









root1 = tkinter.Tk()
root1.geometry('200x200') 

res = Button(root1, text ="Results")
res.pack(side = TOP, pady = 10)

quit = Button(root1, text ="Quit", command = exit)
quit.pack(side = TOP, pady = 10)

def exit():
    root1.quit()

root1.mainloop()

