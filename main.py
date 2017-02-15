# -*- coding: utf-8 -*-
from PyQt4 import QtGui,QtCore
import sys
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

class Traitement_img(QtGui.QWidget):
    def __init__(self):
        super(Traitement_img, self).__init__()

        #---- La figure ou se place les images -----------
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        #------ Les Boutons conecte avec sa fonction ------------
        self.browse = QtGui.QPushButton('Parcourir')
        self.browse.clicked.connect(self.browse_on)
        self.restor = QtGui.QPushButton('Image initale')
        self.restor.clicked.connect(self.rest)
        self.to_nega = QtGui.QPushButton(u'Négative')
        self.to_nega.clicked.connect(self.negative)
        self.to_gris = QtGui.QPushButton('Gris')
        self.to_gris.clicked.connect(self.gray)
        self.to_inverse = QtGui.QPushButton('Inverser')
        self.to_inverse.clicked.connect(self.inv)
        self.to_miroir = QtGui.QPushButton('Miroir')
        self.to_miroir.clicked.connect(self.miroir)
        self.to_rotate = QtGui.QPushButton('rotation')
        self.to_rotate.clicked.connect(self.rotation)
        self.ongle = QtGui.QLineEdit('90')
        self.bruit = QtGui.QPushButton('Ajouter du bruit')
        self.bruit.clicked.connect(self.imnoise)
        self.mean_filt = QtGui.QPushButton('Filte moyenneur')
        self.mean_filt.clicked.connect(self.moy_filter)
        self.mediane_filt = QtGui.QPushButton(u'Filtre médianne')
        self.mediane_filt.clicked.connect(self.mediane_filter)

        #---rotation layout------
        rot = QtGui.QHBoxLayout()
        rot.addWidget(self.ongle)
        rot.addWidget(self.to_rotate)

        #----Bouton Layout_____
        lbo = QtGui.QVBoxLayout()
        lbo.addWidget(self.browse)
        lbo.addWidget(self.restor)
        lbo.addWidget(self.to_nega)
        lbo.addWidget(self.to_gris)
        lbo.addWidget(self.to_inverse)
        lbo.addWidget(self.to_miroir)
        lbo.addLayout(rot)
        lbo.addWidget(self.bruit)
        lbo.addWidget(self.mean_filt)
        lbo.addWidget(self.mediane_filt)

        #------- Le Layout Principal -----
        layout = QtGui.QGridLayout()
        layout.addWidget(self.canvas, 0, 0, 5, 5)
        layout.addLayout(lbo, 0, 5)

        #---------------------------
        self.setLayout(layout)

        self.setWindowTitle(u'Traitement d\'image')
        self.show()

    def negative(self):
        x = self.mat.shape
        try:
            neg = np.zeros(x, dtype=np.uint8)
            for i in range(x[0]):
                for j in range(x[1]):
                    neg[i, j, 0] = 255-self.mat[i, j, 0]
                    neg[i, j, 1] = 255-self.mat[i, j, 1]
                    neg[i, j, 2] = 255-self.mat[i, j, 2]
            self.mat = neg
        except:
            neg = []
            for i in self.mat.ravel():
                neg.append(255-i)
            self.mat = np.array(neg).reshape(x)

        self.print_img(self.mat)

    def gray(self):
        x = self.mat.shape
        gris = np.zeros((x[0], x[1]), dtype=np.uint8)
        if len(x)==3:
            for i in range(x[0]):
                for j in range(x[1]):
                    avg = sum(self.mat[i, j, :])/3
                    gris[i, j] = avg
        self.mat = gris
        self.print_img(self.mat)

    def inv(self):
        x = self.mat.shape
        inver = np.zeros(x, dtype=np.uint8)
        for i in range(1, x[0]):
            inver[i] = self.mat[x[0]-i]
        self.mat = inver
        self.print_img(self.mat)

    def miroir(self):
        x = self.mat.shape
        miroir = np.zeros(x, dtype=np.uint8)
        for i in range(1, x[0]-1):
            for j in range(1, x[1]-1):
                miroir[i, j] = self.mat[i, x[1]-j]
        self.mat = miroir
        self.print_img(self.mat)

    def rotation(self):
        alpha = int(self.ongle.text())
        size = self.mat.shape  #recuperation de la taille de l'image
        B = [0, 0]
        #calcule de la 2eme partie de la regle de translation
        B[0] = (size[0]/2) - ((size[0]/2)*(np.cos(alpha)) + (size[1]/2)*(np.sin(alpha)) )
        B[1] = (size[1]/2) - ((size[1]/2)*(np.cos(alpha)) - (size[0]/2)*(np.sin(alpha)) )
        img = np.zeros(size, dtype=np.uint8)# declaration d'une matrice de sortie
        for i in range(size[0]):
            for j in range(size[1]):
                x = i * np.cos(alpha) + j*np.sin(alpha) + B[0]
                y = j*np.cos(alpha) - i*np.sin(alpha) + B[1]
                if (x < size[0]) & (y < size[1]) & (x>0) & (y>0):
                    img[i, j] = self.mat[x, y]
        self.mat = img.astype('uint8')
        self.print_img(self.mat)

    def mediane_filter(self):
        x = self.mat.shape
        try:
            med = np.zeros(x, dtype=np.uint8)
            for i in range(1, x[0]-1):
                for j in range(1, x[1]-1):
                    r = np.sort(self.mat[i-1:i+1, j-1:j+1, 0].ravel())
                    g = np.sort(self.mat[i-1:i+1, j-1:j+1, 1].ravel())
                    b = np.sort(self.mat[i-1:i+1, j-1:j+1, 2].ravel())
                    med[i, j, 0] = r[len(r)/2]
                    med[i, j, 1] = g[len(g)/2]
                    med[i, j, 2] = b[len(b)/2]
            self.mat = med
        except:
            med = np.zeros(x, dtype=np.uint8)
            for i in range(1, x[0]-1):
                for j in range(1, x[1]-1):
                    r = np.sort(self.mat[i-1:i+1, j-1:j+1, 0].ravel())
                    med[i, j] = r[len(r)/2]
            self.mat = med

        self.print_img(self.mat)

    def moy_filter(self):
        x = self.mat.shape
        masque = np.ones([3, 3])/9.0
        try:
            moy = np.zeros(x, dtype=np.uint8)
            for i in range(1, x[0]-1):
                for j in range(1, x[1]-1):
                    moy[i, j, 0] = sum(((self.mat[i-1:i+2, j-1:j+2, 0].ravel())*(masque.ravel())))
                    moy[i, j, 1] = sum(((self.mat[i-1:i+2, j-1:j+2, 1].ravel())*(masque.ravel())))
                    moy[i, j, 2] = sum(((self.mat[i-1:i+2, j-1:j+2, 2].ravel())*(masque.ravel())))
            self.mat = moy
        except:
            moy = np.zeros(x, dtype=np.uint8)
            for i in range(1, x[0]-1):
                for j in range(1, x[1]-1):
                    moy[i, j] = sum(((self.mat[i-1:i+2, j-1:j+2])*(masque)))
            self.mat = moy

        self.print_img(self.mat)

    def imnoise(self):#fonction qui ajoute un bruit a l'image
        import random as rd
        for i in range(int((self.mat.shape[0]*self.mat.shape[1])*0.1)):
            x = rd.random()*self.mat.shape[0]
            y = rd.random()*self.mat.shape[1]
            self.mat[x, y] = 0 | 1
        self.print_img(self.mat)


    def browse_on(self):
        filename = QtGui.QFileDialog.getOpenFileName(None, 'Open file', "Image(*.jpg *.png *.jpeg)")
        if filename != '':
            filename = str(filename)
            #self.print_img(filename)
            self.mat = plt.imread(filename)
            self.print_img(self.mat)
            self.initail_img = self.mat.copy()

    def rest(self):
        self.mat = self.initail_img
        self.print_img(self.initail_img)

    def print_img(self, matrice):
        try:
            if matrice.shape[2] == 3:
                plt.imshow(matrice)
        except:
            plt.imshow(matrice, cmap=cm.Greys_r)
        plt.axis('off')
        self.canvas.draw()



if __name__=="__main__":
    app = QtGui.QApplication(sys.argv)
    ex = Traitement_img()
    sys.exit(app.exec_())
