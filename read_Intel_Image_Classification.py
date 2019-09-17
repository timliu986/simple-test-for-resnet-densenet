import numpy as np
from PIL import Image
import os



def get_data(type ,data_number):
    X = None
    y = None
    categorie_type = {'buildings' :0, 'forest':1, 'glacier':2, 'mountain':3, 'sea':4, 'street':5}
    t = 6

    for categories in os.listdir('E:/Dataset/Intel_Image_Classification/seg_%s/'%type):
        skip = 0
        n = int(data_number / t)
        k = 0
        while k != n:
            file = os.listdir('E:/Dataset/Intel_Image_Classification/seg_%s/%s/'%(type,categories))[k]
            image = np.array(Image.open('E:/Dataset/Intel_Image_Classification/seg_%s/%s/%s' % (type, categories, file)).resize((50 ,50), Image.BILINEAR).convert('L'))
            if image.shape == (50, 50):
                if X is None:
                    index = 0
                    y = np.empty(data_number)
                    X= np.ndarray([data_number, 50, 50])

                    y[index] = categorie_type[categories]
                    X[index] = np.array(image)
                    k += 1
                else:
                    index += 1
                    y[index] = categorie_type[categories]
                    X[index] = image
                    k += 1
            else :
                k += 1
                skip += 1
        t -= 1
        data_number -= (k-skip)

    return X,y





