from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

x_train = np.array([
    ['Hello i am here'],
    ['Hello i am here 1'],
    ['Hello you am here'],
    ['Hello he am here'],
    ['hi i am there'],
])

x_train = np.array([
    ['a', 'b', 'c'],
    ['a', 'd', 'c'],
    ['e', 'b', 'c'],
])

def one_hot_encode_old(x, k):
    n = len(x)
    #b = np.zeros((n, max(x)+1))
    b = np.zeros((n, k))
    b[np.arange(n), x] = 1
    return b


#from sklearn.preprocessing import LabelBinarizer
#label_binarizer = LabelBinarizer()
#label_binarizer.fit(range(3))

#def one_hot_encode(x):
#    return label_binarizer.transform(x)

def one_hot_encode(x):
    encoder = LabelEncoder()
    x_int = encoder.fit_transform(x.ravel()).reshape(*x.shape)
    return x_int

print(one_hot_encode(x_train))

print(x_train.ravel())



# >>> le = preprocessing.LabelEncoder()
# >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
# LabelEncoder()
# >>> list(le.classes_)
# ['amsterdam', 'paris', 'tokyo']
# >>> le.transform(["tokyo", "tokyo", "paris"])
# array([2, 2, 1]...)
# >>> list(le.inverse_transform([2, 2, 1]))
# ['tokyo', 'tokyo', 'paris']