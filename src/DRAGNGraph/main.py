from stat import filemode
import sys
import os
import PyQt5
from PyQt5 import QtCore
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtGui import *
import stanza
from nltk.tree import Tree
from nltk.tree.prettyprinter import TreePrettyPrinter
from nltk.draw.tree import TreeView
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow,\
     QAction, QGroupBox, QFormLayout, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit,\
        QPushButton, QGridLayout, QTextEdit, QTextBrowser

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
#Image.gs_windows_binary = r'C:\Program Files\gs\gs10.00.0\bin\gswin64c.exe'

class MainWindow(QMainWindow):
    def __init__(self, allowEmptyInput=False, parent=None):
        super().__init__(parent)
        self.createUI()
        self.createMenu()
        self.createLeftHalf()
        self.createRightHalf()
        self.line = 0
        self.aei = allowEmptyInput
        self.nlp = stanza.Pipeline('en', processors='tokenize,lemma,pos,depparse,constituency,ner', use_gpu=False, pos_batch_size=3000)
        #800 x 600

    def createUI(self):
        self.setWindowTitle("DRAGNGraph - Demo")
        self.resize(800, 600)
        self._centralWidget = QWidget()
        self._halfLayout = QGridLayout()
        self._centralWidget.setLayout(self._halfLayout)
        self.setCentralWidget(self._centralWidget)

    def createMenu(self):
        # create menu bar
        menuBar = self.menuBar()
        
        # add menu items
        fileMenu = menuBar.addMenu('File (demo)')
        helpMenu = menuBar.addMenu('Help (demo)')

        #create actions
        visitWebsiteAction = QAction('Visit Our Website (demo)', self)
        fileBugReportAction = QAction('File a Bug Report (demo)', self)
        
        # Add dropdown manu items on the File menu
        fileMenu.addAction('Settings (demo)')
        fileMenu.addAction('Account (demo)')
        fileMenu.addAction('Save as (demo)')
        fileMenu.addAction('Export (demo)')
        
        # Add dropdown menu items on the Help menu
        helpMenu.addAction(visitWebsiteAction)
        helpMenu.addAction(fileBugReportAction)

        # Add 'Follow Us' dropdown menu item on the Help menu
        followUs = helpMenu.addMenu('Follow Us (demo)')

        # Social media actions
        twitterAction = QAction('Twitter (demo)', self)
        githubAction = QAction('GitHub (demo)', self)

        # Add actions
        followUs.addAction(twitterAction)
        followUs.addAction(githubAction)

    def createLeftHalf(self):
        self.createLowerLeft()
        self.createUpperLeft()
        
    def createUpperLeft(self):
        groupBox = QGroupBox()
        groupBox.setTitle('LEFT - UPPER')
        groupBox.setFixedWidth(399)
        groupBox.setFixedHeight(499)
        formLayout = QHBoxLayout()
        self.chatBox = QTextEdit()
        self.chatBox.setReadOnly(True)
        formLayout.addWidget(self.chatBox)


        groupBox.setLayout(formLayout)
        self._halfLayout.addWidget(groupBox, 0, 0, 3, 1)

    def createLowerLeft(self):
        groupBox = QGroupBox()
        groupBox.setTitle('LEFT - LOWER')
        groupBox.setFixedWidth(399)
        groupBox.setFixedHeight(99)
        formLayout = QHBoxLayout()
        
        inputLabel = QLabel("Input:")
        self.inputField = QLineEdit()
        submitButton = QPushButton()
        submitButton.setText("Submit")
        submitButton.clicked.connect(self.submitClicked)
        formLayout.addWidget(inputLabel)
        formLayout.addWidget(self.inputField,4)
        formLayout.addWidget(submitButton)

        groupBox.setLayout(formLayout)
        self._halfLayout.addWidget(groupBox, 3, 0)

    def submitClicked(self):
        print("here0.0")
        if self.inputField.text() != '' or self.aei == True:
            self.chatBox.insertPlainText("[{}] ".format(self.line) + "USER: " + self.inputField.text() + '\n\n')
            self.line += 1
            self.reply(self.inputField.text())
            doc = self.nlp(self.inputField.text())
            print("here0")
            #sentenceConstTrees = []
            #self.treeView.setText(None)
            for sent in doc.sentences:
                #print(TreePrettyPrinter(Tree.fromstring(str(sent.constituency))).text() + '\n')
                import _tkinter
                print(_tkinter.TK_VERSION)
                print(os.getcwd())
                #self.treeView.setText(TreePrettyPrinter(Tree.fromstring(str(sent.constituency))).text() + '\n')
            
                thePath = './src/DRAGNGraph/imgs/tree.ps'

                #TreeView(Tree.fromstring(str(sent.constituency)))._cframe.print_to_file(thePath)
                print("here1")
                #psimg = Image.open(thePath).copy()
                print("here2")
                
                
                #self.fig.setPixmap(QPixmap.fromImage(ImageQt(psimg)))
                
                print("here3")
                #os.system('convert tree.ps tree.png')
                self._halfLayout.update()

            self.inputField.setText(None)

            

    def reply(self, inputText):
        replyText = "This is a default reply."
        self.chatBox.insertPlainText("[{}] ".format(self.line) + "BOT: " + replyText + '\n\n')
        self.line += 1

    def createRightHalf(self):
        self.createLowerRight()
        self.createUpperRight()

    def createUpperRight(self):
        groupBox = QGroupBox()
        groupBox.setTitle('RIGHT - UPPER')
        groupBox.setFixedWidth(399)
        groupBox.setFixedHeight(299)
        formLayout = QHBoxLayout()

        #self.treeView.setFont(QFont('Arial', 10))
        #self.treeView.setReadOnly(True)
        self.fig = QLabel()
        formLayout.addWidget(self.fig)

        groupBox.setLayout(formLayout)
        self._halfLayout.addWidget(groupBox, 0, 1, 2, 1)

    def createLowerRight(self):
        groupBox = QGroupBox()
        groupBox.setTitle('RIGHT - LOWER')
        groupBox.setFixedWidth(399)
        groupBox.setFixedHeight(299)
        formLayout = QVBoxLayout()

        class MplCanvas(FigureCanvasQTAgg):
            def __init__(self, parent=None, width=5, height=4, dpi=100):
                fig = Figure(figsize=(width, height), dpi=dpi)
                self.axes = fig.add_subplot(111)
                super(MplCanvas, self).__init__(fig)
        
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(sc, self)

        formLayout.addWidget(toolbar)
        formLayout.addWidget(sc)

        # Create a placeholder widget to hold our toolbar and canvas.
        groupBox.setLayout(formLayout)
        self._halfLayout.addWidget(groupBox, 2, 1, 2, 1)

        

if __name__ == "__main__":
    print('Loading...')
    app = QApplication(sys.argv)
    mw = MainWindow(False)
    mw.show()
    #app.exec()
    sys.exit(app.exec_())