from sklearn import decomposition as dc
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import os
import pickle
import sklearn.metrics.pairwise as pair


def createDict():
    # Obtain file names of the data in a list
    fileList = os.listdir('data')
    # Used to convert the byte entries to strings(vectorized for Numpy array) of the heads
    def b2s(b):
        return str(b,'utf-8')
    b2sV = np.vectorize(b2s)
    # Creates a dictionary to be filled with {dataset: list(head,data)} entries
    dataDict = {}

    for file in fileList:
        fpath = os.path.join('data',file)
        name = file.rstrip('.headdata')
        dataDict.setdefault(name,[None,None])
        if 'head' in file:
            file = np.loadtxt(fpath, dtype=bytes)
            file = b2sV(file)
            dataDict[name][0] = file
        # Gets rid of NaNs and replaces them with the column's average
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


def pickleArrays():
    # Run first to pickle the dictionary in order to not rebuild it every single time
    dataDict = createDict()
    f = open('modDict','wb')
    pickle.dump(dataDict,f)
    f.close()


def pickleVectors():
    # # Populates a dictionary with vectors representing the truncated matrix with 1 feature
    # Accesses pickled dictionary and instantiates it
    dataDict = pickle.load(open('modDict','rb'))
    # Pickles the vector dictionary
    vDict = {}
    for d in dataDict:
        svd = dc.TruncatedSVD(n_components=1)
        # Transposed in order to create a feature out of all of the rows
        truncated = svd.fit_transform(dataDict[d][1].T)
        vDict[d] = truncated.T
    f = open('vDict','wb')
    pickle.dump(vDict, f)
    f.close()


def loadFiles():
    dataDict = pickle.load(open('modDict', 'rb'))
    vDict = pickle.load(open('vDict', 'rb'))
    return dataDict,vDict


def compare(vector, data, line):
    vector = vDict[vector]
    data = dataDict[data][1][line,:]
    print(pair.pairwise_distances(vector,data,'cosine'))


# Compares and prints out the assigned number of comparisons between the feature vector and the data(starting from assigned line)
def multicompare(vector, data, line, number=20):
    for x in range(number):
        compare(vector,data,line)
        line +=1


# Outputs the closest dialogue and session and the cosine distance in a tuple
def predict(dialogue_session, line):
    lowest = ('x',1)
    data = dataDict[dialogue_session][1][line,:]
    for vector in vDict:
        predictor = vDict[vector]
        if pair.pairwise_distances(predictor,data,'cosine') < lowest[1]:
            lowest = (vector, pair.pairwise_distances(predictor,data,'cosine'))
    return lowest


if __name__ == '__main__':
    dataDict, vDict = loadFiles()
    multicompare('8m5', '7m6', 10, 20)
    print(predict('8m7',20))