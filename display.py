# -*- coding: utf-8 -*-
"""
Created on Thu May 13 06:34:30 2021

@author: karlygash.kussainova
"""

from functionPickle import *

import cv2
import os

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QFileDialog, QAction
from PyQt5.QtGui import QPixmap


pathGlobal = "C:/Users/karlygash.kussainova/Desktop/senior/FYP/finalProject/SavedImg/1.png"


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(575, 402)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.label1l = QtWidgets.QLabel(self.centralwidget)
        self.label1l.setEnabled(True)
        self.label1l.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label1l.setObjectName("label1l")
        self.gridLayout.addWidget(self.label1l, 0, 0, 1, 1)
        self.pushButton1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton1.setObjectName("pushButton1")
        self.gridLayout.addWidget(self.pushButton1, 1, 1, 1, 1)
        self.pushButton2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton2.setObjectName("pushButton2")
        self.gridLayout.addWidget(self.pushButton2, 2, 1, 1, 1)
        self.graphicsView1 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView1.setObjectName("graphicsView1")
        self.gridLayout.addWidget(self.graphicsView1, 3, 0, 1, 1)
        self.tableWidget1 = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget1.setObjectName("tableWidget1")
        self.tableWidget1.setColumnCount(1)
        self.tableWidget1.setRowCount(1)
        self.gridLayout.addWidget(self.tableWidget1, 3, 1, 1, 1)
        
        
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        #self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        
        
        
        self.gridLayout.addWidget(self.progressBar, 2, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 575, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        
        
        MainWindow.setStatusBar(self.statusbar)
        self.pushButton2.clicked.connect(self.on_click)
        
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        
       
        

            
    

        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label1l.setText(_translate(
            "MainWindow", "Cancer Classification App"))
        self.pushButton1.setText(_translate("MainWindow", "Upload image"))
        self.pushButton1.clicked.connect(self.openImage)
        self.pushButton2.setText(_translate("MainWindow", "OK"))
        
        
    def openImage(self):
        imagePath, *_ = QFileDialog.getOpenFileName()
        pixmap = QPixmap(imagePath)
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene = QtWidgets.QGraphicsScene()
        scene.addItem(item)
        self.graphicsView1.setScene(scene)
        img = cv2.imread(imagePath, 1)   
        path = 'C:/Users/karlygash.kussainova/Desktop/senior/FYP/finalProject/SavedImg'
        cv2.imwrite(os.path.join(path , '1.png'), img)
        cv2.waitKey(0)

        
        
    def on_click(self):
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget1.setItem(0, 0, item)
        item.setText(read_img_display(pathGlobal))
        os.remove('C:/Users/karlygash.kussainova/Desktop/senior/FYP/finalProject/SavedImg' + '/' + "1.png")
        self.completed = 0
        while self.completed < 100:
            self.completed += 0.0001
            self.progressBar.setValue(self.completed)
        


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())