# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:20:08 2020

@author: Hamza
"""
#import pySerial
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

def window():
    app=QApplication(sys.argv)
    win=QMainWindow()
    win.setGeometry(100,100,200,200)
    win.setWindowTitle('Testing stuff')
    
    win.show()
    sys.exit(app.exec_())

window()
