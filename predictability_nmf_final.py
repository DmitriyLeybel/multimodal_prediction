from sklearn import decomposition as dc
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import os
import pickle

def createDict():
    # Obtain file names of the data in a list
    fileList = os.listdir('data')
    # Used to convert the byte entries to strings(vectorized for Numpy array) of the heads
    def b2s(b):
        return str(b,'utf-8')
    b2sV = np.vectorize(b2s)
    # Creates a dictionary to be filled with {dataset: tuple(head,data)} entries
    dataDict = {}

    for file in fileList:
        fpath = os.path.join('data',file)
        name = file.rstrip('.headdata')
        dataDict.setdefault(name,[None,None])
        if 'head' in file:
            file = np.loadtxt(fpath, dtype=bytes)
            file = b2sV(file)
            dataDict[name][0] = file
        if 'data' in file:
            file = np.loadtxt(fpath)
            filecp = np.copy(file)
            for i in list(range(file.shape[0])):
                for j in list(range(file.shape[1])):
                    if np.isnan(file[i,j]):
                        column = filecp[:,j]
                        colsum = column[~np.isnan(column)].sum()
                        colavg = colsum/np.count_nonzero(~np.isnan(column))
                        file[i,j] = colsum
            dataDict[name][1] = file
    return dataDict

# # Pickles the dictionary in order to not rebuild it every single time
# dataDict = createDict()
# f = open('modDict','wb')
# pickle.dump(dataDict,f)
# f.close()

# Accesses pickled dictionary and instantiates it
dataDict = pickle.load(open('modDict','rb'))