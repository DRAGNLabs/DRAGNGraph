import traceback, sys
import os

import PyQt5
from PyQt5.QtCore import *
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtGui import *


from PyQt5.QtWidgets import *


#Image.gs_windows_binary = r'C:\Program Files\gs\gs10.00.0\bin\gswin64c.exe'

if __name__ == "__main__":
    print('Loading...')
    print(os.getcwd())
    app = QApplication(sys.argv)
    splash_object = QSplashScreen(QPixmap("./src/DRAGNGraph/imgs/DRAGNlogo.png"))
    splash_object.show()

from stat import filemode
import time

from multiprocessing.dummy import Pool
from DGraph import DGraph

import networkx as nx
from wiki_chase_2 import *
from nltk.tree import Tree
from nltk.tree.prettyprinter import TreePrettyPrinter
from nltk.draw.tree import TreeView
import stanza
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

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
        
        try:
            self.graph = load_object('./src/DRAGNGraph/graphs/Guinea.pkl')
        except:
            self.graph = None
        self.createUI()
        self.createMenu()
        self.createLeftHalf()
        self.createRightHalf()
        #self.createMiddleButton() # FIXME this button works, except for when trying to repopulate the right side after hiding.
        self.line = 0
        self.aei = allowEmptyInput
        self.nlp = stanza.Pipeline('en', processors='tokenize,lemma,pos,depparse,constituency,ner', use_gpu=False, pos_batch_size=3000)
        self.resize(width, height)
        self.tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
        

    def createUI(self):
        self.setWindowTitle("DRAGNGraph - Demo")
        self.setWindowIcon(QIcon("./src/DRAGNGraph/imgs/DRAGNlogo.png"))
        
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

    def createMiddleButton(self, width_mul=1):
        self.hideBtn = QPushButton()
        self.hideBtn.setFixedSize(QSize(30,30))
        
        if width_mul == 1:
            self.hideBtn.setText('>')
        else:
            self.hideBtn.setText('<')
        
        self.hideBtn.clicked.connect(self.hideClicked)
        formLayout = QVBoxLayout()

        formLayout.addWidget(self.hideBtn)
        self._halfLayout.addWidget(self.hideBtn, 0, width_mul*100, 1, 1)

    def hideClicked(self):
        # hide things
        if self.hideBtn.text() == '>':
            self.createUI()
            
            self.createLeftHalf(2, True)
            #self.createRightHalf()
            self.createMiddleButton(2)

        # unhide things
        elif self.hideBtn.text() == '<':
            #self.hideBtn.setText('>')
            self.createUI()
            
            self.createLeftHalf(reLoad=True)
            self.createRightHalf(reLoad=True)
            self.createMiddleButton()
            

    def createLeftHalf(self, width_mul=1, reLoad=False):
        self.createLowerLeft(width_mul, reLoad)
        self.createUpperLeft(width_mul, reLoad)
        
    def createUpperLeft(self, width_mul, reLoad):
        if reLoad:
            self._halfLayout.addWidget(self.groupBox_UL, 0, 0, 8, width_mul*99)
        else:
            groupBox = QGroupBox()
            groupBox.setTitle('LEFT - UPPER')
            groupBox.resize(width_mul*self.width()//2, 8*self.height()//10)
            
            formLayout = QVBoxLayout()
            
            class MplCanvas(FigureCanvasQTAgg):
                def __init__(self, parent, width=4, height=4, dpi=200):
                    self.figure = plt.figure(figsize=(width,height), dpi=dpi)
                    self.figure.set_tight_layout(True)
                    super(MplCanvas, self).__init__(self.figure)
                def draw_graph(self, G, style=None):
                    self.figure.clf()
                    pos = nx.spring_layout(G)
                    nx.draw_networkx(G, pos, with_labels=True)
                    #edge_labels = [e['label'] for e in G.edges]
                    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

            self.sc = MplCanvas(self, width=3, height=3, dpi=100)

            if self.graph != None:
                self.sc.draw_graph(self.graph)

            # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
            toolbar = NavigationToolbar(self.sc, self)
            toolbar.setIconSize(QSize(20,20))
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
            self.groupBox_UL = groupBox
            self._halfLayout.addWidget(self.groupBox_UL, 0, 0, 8, width_mul*99)

    def createLowerLeft(self, width_mul, reLoad):
        if reLoad:
            self._halfLayout.addWidget(self.groupBox_LL, 8, 0, 2, width_mul*99)
        else:
            groupBox = QGroupBox()
            groupBox.setTitle('LEFT - LOWER')
            groupBox.resize(width_mul*self.width()//2, 2*self.height()//10)
            formLayout = QVBoxLayout()
            
            self.chatBox = QTextEdit()
            self.chatBox.setReadOnly(True)
            formLayout.addWidget(self.chatBox)

            inWid = QWidget()
            inputLayout = QHBoxLayout()
            inputLabel = QLabel("Input:")
            self.inputField = QLineEdit()
            self.inputField.returnPressed.connect(self.submitClicked)
            submitButton = QPushButton()
            submitButton.setText("Submit")
            submitButton.clicked.connect(self.submitClicked)
            inputLayout.addWidget(inputLabel)
            inputLayout.addWidget(self.inputField,4)
            inputLayout.addWidget(submitButton)
            inWid.setLayout(inputLayout)
            formLayout.addWidget(inWid)

            groupBox.setLayout(formLayout)
            self.groupBox_LL = groupBox
            self._halfLayout.addWidget(self.groupBox_LL, 8, 0, 2, width_mul*99)

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
            self.chatBox.moveCursor(QTextCursor.End)       

    def reply(self, inputText):
        replyText = "This is a default reply."
        self.chatBox.insertPlainText("[{}] ".format(self.line) + "BOT: " + replyText + '\n\n')
        self.line += 1

    def createRightHalf(self, reLoad=False):
        self.createLowerRight(reLoad)
        self.createUpperRight(reLoad)

    def createUpperRight(self, reLoad):
        if reLoad:
            self._halfLayout.addWidget(self.groupBox_UR, 0, 100, 3, 99)
        else:
            groupBox = QGroupBox()
            groupBox.setTitle('RIGHT - UPPER')
            groupBox.resize(self.width()//2, 3*self.height()//10)
            formLayout = QVBoxLayout()
            #self.fig = QLabel()

            # create coreference resolution text
            coRefWid = QWidget()
            coRefLay = QHBoxLayout()
            self.coRefText = QLabel()
            scroll1 = QScrollArea()

            scroll1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll1.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            scroll1.setWidgetResizable(True)
            scroll1.setWidget(coRefWid)

            self.coRefText.setText('<font color="purple">Trevor</font><font color="black"> is making this. </font><font color="purple">Trevor </font><font color="black">said that this is where Coreference Resolution would be.</font>')
            coRefLay.addWidget(self.coRefText)
            coRefWid.setLayout(coRefLay)
            formLayout.addWidget(scroll1)

            # create node layout that will get displayed for insertion
            class MplCanvas2(FigureCanvasQTAgg):
                def __init__(self, parent, width=4, height=4, dpi=200):
                    self.figure = plt.figure(figsize=(width,height), dpi=dpi)
                    self.figure.set_tight_layout(True)
                    super(MplCanvas2, self).__init__(self.figure)
                def draw_graph(self, G, style=None):
                    self.figure.clf()
                    pos = nx.spring_layout(G)
                    nx.draw_networkx(G, pos, with_labels=True)
                    edge_labels = nx.get_edge_attributes(G, 'label')
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

            #self.nodeInWid = MplCanvas2(self, width=3, height=1, dpi=75)
            # create nodes
            #newGraph = DGraph()
            #N = ["Trevor", "this", "CoRefRes"]
            #newGraph.add_multiple_nodes(N)
            #E = [("Trevor", "this", {"label":"made"}),("Trevor", "CoRefRes", {"label":"said"})]
            #newGraph.add_multiple_edges(E)
            #self.nodeInWid.draw_graph(newGraph.G, 'spring')

            #toolbar = NavigationToolbar(self.nodeInWid, self)
            #toolbar.setIconSize(QSize(20,20))
            
            #formLayout.addWidget(self.nodeInWid)
            #formLayout.addWidget(toolbar)

            groupBox.setLayout(formLayout)
            self.groupBox_UR = groupBox
            self._halfLayout.addWidget(self.groupBox_UR, 0, 100, 3, 99)

    def createLowerRight(self, reLoad):
        if reLoad:
            self._halfLayout.addWidget(self.groupBox_LR, 3, 100, 7, 99)
        else:
            groupBox = QGroupBox()
            groupBox.setTitle('RIGHT - LOWER')
            groupBox.resize(self.width()//2, 7*self.height()//10)
            formLayout = QVBoxLayout()

            # PLACE CONTENTS HERE

            groupBox.setLayout(formLayout)
            self.groupBox_LR = groupBox
            self._halfLayout.addWidget(self.groupBox_LR, 3, 100, 7, 99)

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
        time.sleep(2)
        self.update()
        # create new graph
        title = extract_and_save(urls, self.tokenizer, self.model)
        new_graph = load_object('./src/DRAGNGraph/graphs/{}.pkl'.format(title))
        # set new graph to self.sc
        self.sc.draw_graph(new_graph)

if __name__ == "__main__":
    

    
    mw = MainWindow(False)
    splash_object.close()
    mw.show()
    #app.exec()
    sys.exit(app.exec_())