import matplotlib.pyplot as plt
import numpy

def negative(image):
    x = image.shape
    #print type(image[0,0,0])
    #print type(image)
    for i in range(x[0]):
        for j in range(x[1]):
            image[i,j,0]= 255-image[i,j,0]
            image[i,j,1]= 255-image[i,j,1]
            image[i,j,2]= 255-image[i,j,2]
    return image

def gray_avg(image):
    x = image.shape
    for i in range(x[0]):
        for j in range(x[1]):
            avg = sum(image[i,j,:])/3
            image[i,j]= (avg,avg,avg)
            #image[i,j,1]=avg
            #image[i,j,2]=avg
    return image

def gray_709(image):
    x = image.shape
    #print type(image[0,0,0])
    #print type(image)
    for i in range(x[0]):
        for j in range(x[1]):
            gris = int(round( float(image[i,j,0])*0.2125 + float(image[i,j,1])*0.7154 + float(image[i,j,2])*0.0721 ))
            image[i,j]= (gris,gris,gris)
            #image[i,j,1]=avg
            #image[i,j,2]=avg
    return image

def gray_tmax(image):
    x = image.shape
    #print type(image[0,0,0])
    #print type(image)
    for i in range(x[0]):
        for j in range(x[1]):
            gris = int(round( float(image[i,j,0])*0.27 + float(image[i,j,1])*0.36 + float(image[i,j,2])*0.37 ))
            image[i,j]= (gris,gris,gris)
            #image[i,j,1]=avg
            #image[i,j,2]=avg
    return image

def gray_NTSC(image):
    x = image.shape
    image.astype(float)
    #print type(image[0,0,0])
    #print type(image)
    for i in range(x[0]):
        for j in range(x[1]):
            gris = (round( (image[i,j,0])*0.2989 + (image[i,j,1])*0.5870 + (image[i,j,2])*0.1140 ))
            image[i,j]= (gris,gris,gris)
            #image[i,j,1]=avg
            #image[i,j,2]=avg
    image.astype(numpy.uint8)
    return image

def histograme(image):
    hist = [0 for i in range(256)]
    x= image.shape
    for i in range(x[0]):
        for j in range(x[1]):
            hist[int(image[i,j,0])]+=1
    return hist

def egalisation_hist(image,hist):
    ega_hist = [0 for i in range(256)]
    size = image.shape
    size = size[0]*size[1]
    cd =cdf(hist,size)
    for i in range(256):
        ega_hist[int(round(255 * cd[i]))]+= hist[i]
        #print "ega histo ", i , "-->", ega_hist[i],"\n"
    return ega_hist

def egalisation_image(image , hist ):
    x = image.shape
    sz = x[0] * x[1]
    cd = cdf(hist,sz)
    for i in range(x[0]):
        for j in range(x[1]):
            t = round(255 * cd[int(image[i,j,0])])
            #print " eqa -----> ",t,"\n"
            image[i,j] = ( t , t , t )
    return image

def cdf(hist,size):
    prob_c = 0.0
    cd =[0 for i in range(256)]
    for i in range(256):
        prob_c += float(hist[i]) / float(size)
        cd[i] = prob_c
    #print cd
    return cd

def RGB_TO_NTSC(image):
    size = image.shape
    img = numpy.zeros(size,dtype=numpy.float)
    for i in range(size[0]):
        for j in range(size[1]):
            img[i,j,0] = (round(image[i,j,0] * 0.299 + image[i,j,1] * 0.587 + image[i,j,2] * 0.114))
            img[i,j,1] = (round(image[i,j,0] * 0.596 - image[i,j,1] * 0.274 - image[i,j,2] * 0.322))
            img[i,j,2] = (round(image[i,j,0] * 0.212 - image[i,j,1] * 0.523 + image[i,j,2] * 0.311))
    return img

def NTSC_TO_RGB(image):
    size = image.shape
    image.astype(float)
    for i in range(size[0]):
        for j in range(size[1]):
            r = int(float(image[i,j,0]) + float(image[i,j,1])* 0.9563 + float(image[i,j,2]) * 0.6210)
            g = int(float(image[i,j,0]) - float(image[i,j,1])* 0.2721 - float(image[i,j,2]) * 0.6474)
            b = int(float(image[i,j,0]) - float(image[i,j,1])* 1.1070 + float(image[i,j,2]) * 1.7046)
            image[i,j] = (r , g , b)
    image.astype(numpy.uint8)
    return image

def miroir(image):
    x=image.shape
    mat = numpy.zeros(x,dtype=numpy.uint8)
    for i in range(1,x[0]-1):
        for j in range(1,x[1]-1):
            mat[i,j] = image[i,x[1]-j]
    return mat

def inverse(image):
    x=image.shape
    mat = numpy.zeros(x,dtype=numpy.uint8)
    for i in range(1,x[0]):
        mat[i] = image[x[0]-i]
    return mat

def rotation_90_gauche(image):
    x = image.shape
    mat = numpy.zeros((x[1],x[0],3),dtype=numpy.uint8)
    for j in range(x[0]-1,-1,-1):
        for i in range(0,x[1]):
            mat[i,j] = image[j,i]
    return mat

def rotation_90_droite(image):
    mat = rotation_90_gauche(image)
    return miroir(mat)
########################################################################
def baguette_magique(image,seuil,lig,col):
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            if (image[lig+i,col+j,0] > image[lig+i,col+j,0]+seuil) or (image[lig+col+i,j,0] < image[lig+col+i,j,0]-seuil):
                image[lig+i,col+j,0] = image[0,0,0]
            if (image[lig+i,col+j,1] > image[lig+i,col+j,1]+seuil) or (image[lig+col+i,j,1] < image[lig+i,col+j,1]-seuil):
                image[lig+i,col+j,1] = image[0,0,1]
            if (image[lig+i,col+j,2] > image[lig+i,col+j,2]+seuil) or (image[lig+i,col+j,2] < image[lig+i,col+j,2]-seuil):
                image[lig+i,col+j,2] = image[0,0,2]

def masque_baguette(image):
    x=image.shape
    for i in range(2,x[0]-2):
        for j in range(2,x[1]-2):
            baguette_magique(image,100,i,j)
    return image
###Filtre Mediane#####################################################################
def mediane(image_gris,masque,ligne,colone):
    ar = []
    for i in range(-masque,masque+1):
        for j in range(-masque,masque+1):
            ar.append(image_gris[ligne+i,colone+j,0])
    ar.sort()
    #print ar
    #print ar[len(ar)/2]
    return (ar[len(ar)/2],ar[len(ar)/2],ar[len(ar)/2])

def mediane_all(img_gri):
    x=img_gri.shape
    for i in range(2,x[0]-2):
        for j in range(2,x[1]-2):
            img_gri[i,j] = mediane(img_gri,1,i,j)
    return img_gri

###Filtre Moyeneur#####################################################################

def moy_all(img_gri):
    x = img_gri.shape
    img_gri.astype(float)
    mat = numpy.zeros(x,dtype=int)
    for ligne in range(1,x[0]-1):
        for colone in range(1,x[1]-1):
            a=0
            for i in range(-1,1):
                for j in range(-1,1):
                    a += img_gri[ligne+i,colone+j,0]
            a= a/9.
            mat[ligne,colone] = (a, a, a)
    mat.astype(numpy.uint8)
    return mat

########################################################################
def resize(image, facteur,factCol):
    x = image.shape
    l = int(x[0]*facteur) , int(x[1]*factCol) , x[2]
    mat = numpy.zeros(l,dtype=numpy.uint8)
    for i in range(0,x[0],2):
        for j in range(0,x[1],2):
            mat[i*facteur,j*factCol]=image[i,j]
    return mat

def rotation(img,alpha):
    size = img.shape  #recuperation de la taille de l'image
    B =[0,0]
    #calcule de la 2eme partie de la regle de translation
    B[0] = (size[0]/2) - ((size[0]/2)*(numpy.cos(alpha)) + (size[1]/2)*(numpy.sin(alpha)) )
    B[1] = (size[1]/2) - ((size[1]/2)*(numpy.cos(alpha)) - (size[0]/2)*(numpy.sin(alpha)) )
    mat = numpy.zeros(size,dtype=numpy.uint8) # declaration d'un matrice de sortie
    for i in range(size[0]):
        for j in range(size[1]):
            x = i * numpy.cos(alpha) + j*numpy.sin(alpha) + B[0]
            y = j*numpy.cos(alpha) - i*numpy.sin(alpha) + B[1]
            if (x < size[0]) & (y < size[1]) & (x>0) & (y>0):
                mat[i,j]=img[x,y]
    return mat

def rgb_YCbCr(image):
    size = image.shape
    image.astype(float)
    mat = numpy.zeros(size,numpy.uint8)
    for i in range(size[0]):
        for j in range(size[1]):
            mat[i,j,0] = int(0.299*image[i,j,0] + 0.587*image[i,j,1] + 0.114*image[i,j,2])
            mat[i,j,1] = int(128 - 0.169*image[i,j,0] - 0.332*image[i,j,1] + 0.5*image[i,j,2])
            mat[i,j,2] = int(128 + 0.5*image[i,j,0] - 0.419*image[i,j,1] - 0.081*image[i,j,2])
    return mat

def filtre_lin(image, masque):
    mat = numpy.zeros(image.shape, 'uint8')
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            mat[i, j, 0] = sum(((image[i-1:i+2, j-1:j+2, 0].ravel())*(masque.ravel())))
            mat[i, j, 1] = sum(((image[i-1:i+2, j-1:j+2, 1].ravel())*(masque.ravel())))
            mat[i, j, 2] = sum(((image[i-1:i+2, j-1:j+2, 2].ravel())*(masque.ravel())))
    #print sum((image[0:3,0:3,0].ravel())*(masque.ravel())).astype(numpy.uint8)
    #mat.astype('uint8')
    return mat

def mediane_filter(image):
    mat = numpy.zeros(image.shape, numpy.uint8)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            r = numpy.sort(image[i-1:i+1, j-1:j+1, 0].ravel())
            g = numpy.sort(image[i-1:i+1, j-1:j+1, 1].ravel())
            b = numpy.sort(image[i-1:i+1, j-1:j+1, 2].ravel())
            mat[i, j, 0] = r[len(r)/2]
            mat[i, j, 1] = g[len(g)/2]
            mat[i, j, 2] = b[len(b)/2]
    return mat

def imnoise(image):#fonction qui ajoute un bruit a l'image
    import random as rd
    for i in range(int((image.shape[0]*image.shape[1])*0.1)):
        x = rd.random()*image.shape[0]
        y = rd.random()*image.shape[1]
        image[x,y] = 0 | 1
    return image

if __name__ == '__main__':
    img = plt.imread('biche.jpg')
    #img = gray_avg(img)
    img =imnoise(img)
    plt.subplot(1,2,1)
    plt.imshow(img)
    """ycbcr = rgb_YCbCr(img)
    print ycbcr
    plt.imshow(ycbcr)"""

    moy = numpy.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.
    pass_haut = numpy.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 'float')
    sobel = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 'float')
    sobel2 = numpy.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 'float')
    #print moy
    #print fill
 
    moyn = mediane_filter(img.copy())
    plt.subplot(1, 2, 2)
    plt.imshow(moyn)

    """hist=histograme(img)
    plt.subplot(2,2,2)
    plt.xlim(0,255)
    plt.plot(hist)

    egaimg = egalisation_image(img.copy(), hist)
    plt.subplot(2,2,3)
    plt.imshow(egaimg)

    egahiust=egalisation_hist(img,hist)
    plt.subplot(2,2,4)
    plt.xlim(0,255)
    plt.plot(egahiust)"""

    plt.show()
