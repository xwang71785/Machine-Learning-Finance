# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:31:15 2017

Programming Python
@author: wangx3
"""
"""
# List
bob = ['Bob Smith', 42, 30000, 'software']
sue = ['Sue Jones', 45, 40000, 'hardware']
last_name = bob[0].split()[-1]    # method split()

people = [bob, sue]
for person in people:
    print(person[0].split()[1])
pays = [person[2] for person in people]
print(pays)

salaries = list(map((lambda x: x[2]), people))
# list（）强制转换成List格式
# map（）只是生成一个内存实例
# lambda 定义一个计算公式
print(salaries)

people.append(['Tom', 50, 0, None])
"""

# Dictionary
# bob = {'name': 'Bob Smith', 'age': 42, 'pay': 30000, 'job': 'dev'}
# sue = {'name': 'Sue Jones', 'age': 45, 'pay': 40000, 'job': 'hdw'}

# Another way to define the Dictionary
# bob = dict(name='Bob Smith', age=42, pay=30000, job='dev')
# sue = dict(name='Sue Jones', age=45, pay=40000, job='hdw')
'''
# OOP
class Person:
    def __init__(self, name, age, pay=0, job=None):
        self.name = name
        self.age = age
        self.pay = pay
        self.job = job
    def lastName(self):
        return self.name.split()[-1]
    def giveRaise(self, percent):
        self.pay *= (1.0 + percent)
    def __str__(self):
        return ('<%s => %s: %s, %s>' % 
               (self.__class__.__name__, self.name, self.job, self.pay))

class Manager(Person):    # inherited from parent class
    def __init__(self, name, age, pay):
        # call back the superclass' method directly
        Person.__init__(self, name, age, pay, 'manager')
    def giveRaise(self, percent, bonus=0.1):
        Person.giveRaise(self, percent + bonus) 
    
if __name__ == '__main__':
    bob = Person('Bob Smith', 42, 30000, 'software')
    sue = Person('Sue Jones', 45, 40000, 'hardware')
    tom = Manager(name='Tom Doe', age=50, pay=50000)
    team = [bob, sue, tom]
    for obj in team:
        obj.giveRaise(.15)
        print(obj)
 '''       
'''  
import tkinter as tk # get widget classes
import tkinter.messagebox as tkm # get standard dialogs

def notdone():
    tkm.showerror('Not implemented', 'Not yet available')
    
def makemenu(win):
    top = tk.Menu(win) # win=top-level window
    win.config(menu=top) # set its menu option
    file = tk.Menu(top)
    file.add_command(label='New...', command=notdone, underline=0)
    file.add_command(label='Open...', command=notdone, underline=0)
    file.add_command(label='Quit', command=win.quit, underline=0)
    top.add_cascade(label='File', menu=file, underline=0)
    
    edit = tk.Menu(top, tearoff=False)
    edit.add_command(label='Cut', command=notdone, underline=0)
    edit.add_command(label='Paste', command=notdone, underline=0)
    edit.add_separator()
    top.add_cascade(label='Edit', menu=edit, underline=0)
    
    submenu = tk.Menu(edit, tearoff=True)
    submenu.add_command(label='Spam', command=win.quit, underline=0)
    submenu.add_command(label='Eggs', command=notdone, underline=0)
    edit.add_cascade(label='Stuff', menu=submenu, underline=0)
    
if __name__ == '__main__':
    root = tk.Tk() # or Toplevel()
    root.title('menu_win') # set window-mgr info
    makemenu(root) # associate a menu bar
    msg = tk.Label(root, text='Window menu basics') # add something below
    msg.pack(expand=tkm.YES, fill=tk.BOTH)
    msg.config(relief=tk.SUNKEN, width=40, height=7, bg='beige')
    root.mainloop()
 '''

from tkinter import *
def frame(root, side=TOP, **extras):
    widget = Frame(root)
    widget.pack(side=side, expand=YES, fill=BOTH)
    if extras: widget.config(**extras)
    return widget
def label(root, side, text, **extras):
    widget = Label(root, text=text, relief=RIDGE) # default config
    widget.pack(side=side, expand=YES, fill=BOTH) # pack automatically
    if extras: widget.config(**extras) # apply any extras
    return widget
def button(root, side, text, command, **extras):
    widget = Button(root, text=text, command=command)
    widget.pack(side=side, expand=YES, fill=BOTH)
    if extras: widget.config(**extras)
    return widget
def entry(root, side, linkvar, **extras):
    widget = Entry(root, relief=SUNKEN, textvariable=linkvar)
    widget.pack(side=side, expand=YES, fill=BOTH)
    if extras: widget.config(**extras)
    return widget
if __name__ == '__main__':
    app = Tk()
    frm = frame(app, TOP) # much less code required here!
    label(frm, LEFT, 'SPAM')
    button(frm, BOTTOM, 'Press', lambda: print('Pushed'))
    mainloop()

    