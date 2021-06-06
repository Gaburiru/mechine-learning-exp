import numpy
import pickle

class NPDFeature():
    """It is a tool class to extract the NPD features.

    Attributes:
        image: A two-dimension ndarray indicating grayscale image.
        n_pixels: An integer indicating the number of image total pixels.
        features: A one-dimension ndarray to store the extracted NPD features.
    """
    __NPD_table__ = None

    def __init__(self, image):
        '''Initialize NPDFeature class with an image.'''
        if NPDFeature.__NPD_table__ is None:
            NPDFeature.__NPD_table__ = NPDFeature.__calculate_NPD_table()
        assert isinstance(image, numpy.ndarray)
        self.image = image.ravel()
        self.n_pixels = image.size
        self.features = numpy.empty(shape=self.n_pixels * (self.n_pixels - 1) // 2, dtype=float)

    def extract(self):
        '''Extract features from given image.

        Returns:
            A one-dimension ndarray to store the extracted NPD features.
        '''
        count = 0
        for i in range(self.n_pixels - 1):
            for j in range(i + 1, self.n_pixels, 1):
                self.features[count] = NPDFeature.__NPD_table__[self.image[i]][self.image[j]]
                count += 1
        return self.features

    @staticmethod
    def __calculate_NPD_table():
        '''Calculate all situations table to accelerate feature extracting.'''
        print("Calculating the NPD table...")
        table = numpy.empty(shape=(1 << 8, 1 << 8), dtype=float)
        for i in range(1 << 8):
            for j in range(1 << 8):
                if i == 0 and j == 0:
                    table[i][j] = 0
                else:
                    table[i][j] = (i - j) / (i + j)
        return table

test_img=[]
import os
from PIL import Image
import torchvision.transforms as transforms
def read_img(dirPath):
    """
    :param dirPath:文件夹路径
    :return :

    """
    for file in os.listdir(dirPath):
        if os.path.isfile(os.path.join(dirPath,file))==True:
            c=os.path.basename(file)
            name=dirPath+'/'+c
            img=Image.open(name)
            img_transforms=transforms.Compose([
                transforms.Grayscale(1)
            ])
            img=img_transforms(img)
            img=img.resize((24,24),Image.ANTIALIAS)
            test_img.append(img)
          
read_img('D:/mlexp3/ML2018-lab-03-master/datasets/original/face')
read_img('D:/mlexp3/ML2018-lab-03-master/datasets/original/nonface')
y=numpy.ones(500)
for i in range(500):
	y=numpy.append(y,-1)
y.reshape(1000,1)

model=[]
for i in range(0,y.size):               
       X_=numpy.array(test_img[i])
       object=NPDFeature(X_)
       var=object.extract() 
       model.append(var)
mat=numpy.array(model)
mat.reshape(1000,-1)
data=numpy.column_stack((y,mat))
       
def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

save(data,'feature.txt')
'''
def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
data=load('feature.json')
'''