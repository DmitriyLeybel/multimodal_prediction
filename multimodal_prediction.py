from sklearn import decomposition as dc
import numpy as np

modalitiesArray = np.loadtxt('4m1.data')
valueNamesArray = np.loadtxt('4m1.head', dtype=bytes)

def b2s(b):
    return str(b,'utf-8')
b2sV = np.vectorize(b2s)
valueNamesArray = b2sV(valueNamesArray)

print(modalitiesArray.shape)
print(valueNamesArray.shape)



# nnmf = dc.NMF(n_components=1)
# W = nnmf.fit_transform(modalitiesArray)
# H = nnmf.components_
# predictedVNA = np.dot(W,H)



