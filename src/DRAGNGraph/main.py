from stat import filemode
import time
import traceback, sys
import os
from multiprocessing.dummy import Pool

import networkx as nx
from wiki_chase_2 import *

import PyQt5
from PyQt5.QtCore import *
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

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    #finished = pyqtSignal()
    #progress = pyqtSignal(int)

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        #self.kwargs['progress_callback'] = self.signals.progress
    
    @pyqtSlot()
    def run(self):
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class MainWindow(QMainWindow):
    def __init__(self, allowEmptyInput=False, parent=None, width=800, height=600):
        super().__init__(parent)
        
        self.graph = load_object('./src/DRAGNGraph/graphs/Guinea.pkl')
        
        self.createUI()
        self.createMenu()
        self.createLeftHalf()
        self.createRightHalf()
        self.line = 0
        self.aei = allowEmptyInput
        self.nlp = stanza.Pipeline('en', processors='tokenize,lemma,pos,depparse,constituency,ner', use_gpu=False, pos_batch_size=3000)
        self.resize(width, height)
        self.tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
        #800 x 600

    def createUI(self):
        self.setWindowTitle("DRAGNGraph - Demo")
        #self.resize(800, 600)
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
        groupBox.resize(self.width()//2, 8*self.height()//10)
        formLayout = QHBoxLayout()
        self.chatBox = QTextEdit()
        self.chatBox.setReadOnly(True)
        formLayout.addWidget(self.chatBox)


        groupBox.setLayout(formLayout)
        self._halfLayout.addWidget(groupBox, 0, 0, 8, 1)

    def createLowerLeft(self):
        groupBox = QGroupBox()
        groupBox.setTitle('LEFT - LOWER')
        groupBox.resize(self.width()//2, 2*self.height()//10)
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
        self._halfLayout.addWidget(groupBox, 8, 0, 2, 1)

    def submitClicked(self):
        #print("here0.0")
        if self.inputField.text() != '' or self.aei == True:
            self.chatBox.insertPlainText("[{}] ".format(self.line) + "USER: " + self.inputField.text() + '\n\n')
            self.line += 1
            self.reply(self.inputField.text())
            doc = self.nlp(self.inputField.text())
            #print("here0")
            #sentenceConstTrees = []
            #self.treeView.setText(None)
            for sent in doc.sentences:
                #print(TreePrettyPrinter(Tree.fromstring(str(sent.constituency))).text() + '\n')
                import _tkinter
                #print(_tkinter.TK_VERSION)
                #print(os.getcwd())
                #self.treeView.setText(TreePrettyPrinter(Tree.fromstring(str(sent.constituency))).text() + '\n')
            
                thePath = './src/DRAGNGraph/imgs/tree.ps'

                #TreeView(Tree.fromstring(str(sent.constituency)))._cframe.print_to_file(thePath)
                #print("here1")
                #psimg = Image.open(thePath).copy()
                #print("here2")
                
                
                #self.fig.setPixmap(QPixmap.fromImage(ImageQt(psimg)))
                
                #print("here3")
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
        groupBox.resize(self.width()//2, 3*self.height()//10)
        formLayout = QHBoxLayout()
        self.fig = QLabel()
        formLayout.addWidget(self.fig)

        groupBox.setLayout(formLayout)
        self._halfLayout.addWidget(groupBox, 0, 1, 3, 1)

    def createLowerRight(self):
        groupBox = QGroupBox()
        groupBox.setTitle('RIGHT - LOWER')
        groupBox.resize(self.width()//2, 7*self.height()//10)
        formLayout = QVBoxLayout()

        class MplCanvas(FigureCanvasQTAgg):
            def __init__(self, parent, width=4, height=4, dpi=200):
                self.figure = plt.figure(figsize=(width,height), dpi=dpi)
                self.figure.set_tight_layout(True)
                super(MplCanvas, self).__init__(self.figure)
            def draw_graph(self, G):
                self.figure.clf()
                nx.draw_networkx(G)
                
                

        print(self.graph)
        self.sc = MplCanvas(self, width=3, height=3, dpi=100)
        #sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        self.sc.draw_graph(self.graph)

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.sc, self)

        formLayout.addWidget(toolbar)
        formLayout.addWidget(self.sc)

        wiki_widget = QWidget()
        wiki_widget.setMaximumHeight(self.height()//10)
        horz_form = QHBoxLayout()
        wikiLabel = QLabel("Wiki URL:")
        self.wikiInput = QLineEdit()
        self.wikiSubmit = QPushButton()
        self.wikiSubmit.setText("Wiki")
        self.wikiSubmit.clicked.connect(self.wikiClicked)
        horz_form.addWidget(wikiLabel)
        horz_form.addWidget(self.wikiInput)
        horz_form.addWidget(self.wikiSubmit)
        wiki_widget.setLayout(horz_form)

        formLayout.addWidget(wiki_widget)


        # Create a placeholder widget to hold our toolbar and canvas.
        groupBox.setLayout(formLayout)
        self._halfLayout.addWidget(groupBox, 3, 1, 7, 1)

    def wikiClicked(self):
        if self.wikiInput.text() != '':
            self.thr = QThreadPool()
            self.worker = Worker(self.asyncgraph)
            #self.worker.signals.result.connect(self.asyncdone)
            self.worker.signals.finished.connect(self.asyncdone)
            self.thr.start(self.worker)

    def asyncdone(self):
        print('ASYNC DONE!')
        self.sc.draw()
        self.wikiInput.setText(None)
        self.wikiSubmit.setEnabled(True)
        self.wikiInput.setEnabled(True)
        self.update()

    def asyncgraph(self):
        # get url
        urls = [""]
        urls[0] = self.wikiInput.text()
        self.wikiInput.setText("            ......LOADING......")
        self.wikiInput.setEnabled(False)
        self.wikiSubmit.setEnabled(False)
        self.update()
        # create new graph
        title = extract_and_save(urls, self.tokenizer, self.model)
        new_graph = load_object('./src/DRAGNGraph/graphs/{}.pkl'.format(title))
        # set new graph to self.sc
        self.sc.draw_graph(new_graph)

if __name__ == "__main__":
    print('Loading...')
    #from tkinter import * 
    #from tkinter.ttk import *
    # creating tkinter window
    #root = Tk()
    # getting screen's height in pixels
    #height = root.winfo_screenheight()
    # getting screen's width in pixels
    #width = root.winfo_screenwidth()
    #print("\n width x height = %d x %d (in pixels)\n" %(width, height))

    app = QApplication(sys.argv)
    mw = MainWindow(False)
    mw.show()
    #app.exec()
    sys.exit(app.exec_())