from sklearn import decomposition as dc
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import os
import pickle
import sklearn.metrics.pairwise as pair
from sklearn.preprocessing import Imputer, MinMaxScaler
import re
import matplotlib.pyplot as plt


def createDict(dataDir = 'data'):
    # Obtain file names of the data in a list
    fileList = os.listdir(dataDir)
    # Used to convert the byte entries to strings(vectorized for Numpy array) of the heads
    def b2s(b):
        return str(b,'utf-8')
    b2sV = np.vectorize(b2s)
    # Creates a dictionary to be filled with {dataset: list(head,data)} entries
    dataDict = {}

    for file in fileList:
        fpath = os.path.join(dataDir,file)
        name = file.rstrip('.headdata')
        dataDict.setdefault(name,[None,None])
        if 'head' in file:
            file = np.loadtxt(fpath, dtype=bytes)
            file = b2sV(file)
            dataDict[name][0] = file
        # Gets rid of NaNs and replaces them with the column's average
        if 'data' in file:
            file = np.loadtxt(fpath)
            # filecp = np.copy(file)
            # for i in list(range(file.shape[0])):
            #     for j in list(range(file.shape[1])):
            #         if np.isnan(file[i,j]):
            #             column = filecp[:,j]
            #             colsum = column[~np.isnan(column)].sum()
            #             colavg = colsum/np.count_nonzero(~np.isnan(column))
            #             file[i,j] = colsum

            imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
            file = imp.fit_transform(file)
            mmax = MinMaxScaler()
            file = mmax.fit_transform(file)
            dataDict[name][1] = file
    return dataDict


def createDictV2(dataDir = 'data'):
    # Obtain file names of the data in a list
    fileList = os.listdir(dataDir)
    # Used to convert the byte entries to strings(vectorized for Numpy array) of the heads
    def b2s(b):
        return str(b,'utf-8')
    b2sV = np.vectorize(b2s)
    # Creates a dictionary to be filled with {session: list(head,listofdata)} entries
    dataDict = {}
    for file in fileList:
        fpath = os.path.join(dataDir,file)
        # fname = file.rstrip('.headdata')
        namestring = re.findall('[0-9]+',file)[1]
        dataDict.setdefault(namestring,[None,[]])
        # Adds only one head(some heads may have different amount of columns)
        if 'head' in file and dataDict[namestring][0] is None:
            dataDict[namestring][0] = b2sV(np.loadtxt(fpath, dtype=bytes))
        if 'data' in file:
            dataDict[namestring][1].append(np.loadtxt(fpath))
    # Merges all of the arrays in every dialogue dictionary entry into one
    for k in dataDict:
        dataDict[k][1] = np.row_stack((c for c in dataDict[k][1]))
    # Fills in the averages of each dialogue data array into the NaN spots
    for k in dataDict:
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        dataDict[k][1] = imp.fit_transform(dataDict[k][1])
        mmax = MinMaxScaler()
        dataDict[k][1] = mmax.fit_transform(dataDict[k][1])
    # Gets rid of the first two columns - dialogue and session
    for k in dataDict:
        dataDict[k][0] = dataDict[k][0][2:]
        dataDict[k][1] = dataDict[k][1][:,2:]
    return dataDict


def pickleArrays():
    # Run first to pickle the dictionary in order to not rebuild it every single time
    print('Constructing dictionary')
    dataDict = createDictV2()
    print('Dictionary constructed. Now, pickling dictionary.')
    with open('modDict','wb') as f:
        pickle.dump(dataDict,f)
    print('Pickling complete')


def pickleVectors():
    # # Populates a dictionary with vectors representing the truncated matrix with 1 feature
    # Accesses pickled dictionary and instantiates it
    print('Pickling vectors')
    dataDict = pickle.load(open('modDict','rb'))
    # Pickles the vector dictionary
    vDict = {}
    for d in dataDict:
        svd = dc.TruncatedSVD(n_components=1)
        # Transposed in order to create a feature out of all of the rows
        truncated = svd.fit_transform(dataDict[d][1].T)
        vDict[d] = truncated.T
    with open('vDict','wb') as f:
        pickle.dump(vDict, f)
    print('Finished pickling vectors')

def loadFiles():
    dataDict = pickle.load(open('modDict', 'rb'))
    vDict = pickle.load(open('vDict', 'rb'))
    return dataDict,vDict


def compare(vector, data, line):
    vector = vDict[vector]
    data = np.reshape(dataDict[data][1][line,:],(1,-1)) # Reshape to create 2d array(avoids validation errors)
    print(pair.pairwise_distances(vector,data,'cosine'))


def compareV2(vector, data, width=20, seedNum=0):
    np.random.seed(seedNum)
    vector = vDict[vector]
    dataSize = dataDict[data][1].shape[0]
    randomMid = np.random.random_integers(0,dataSize-1)
    slice = dataDict[data][1][randomMid-width/2:randomMid+width/2, :]
    svd = dc.TruncatedSVD(n_components=1)
    truncated = svd.fit_transform(slice.T).T
    return truncated, pair.pairwise_distances(vector, truncated, 'cosine')

# Compares and prints out the assigned number of comparisons between the feature vector and the data(starting from assigned line)
def multicompare(vector, data, line, number=20):
    for x in range(number):
        compare(vector,data,line)
        line +=1

def multiCompareV2(data, width=20, seedNum=0):
    print('Comparing a random portion of dialogue {data} with width {range} to the dialogue vectors \n'.format(data=data,
                                                                                                            range=width), 40*'*-')
    distances = []
    for v in vDict:
        trunc, comp = compareV2(v, data, width, seedNum=seedNum)
        distance = float(comp)
        distances.append((v,distance,trunc))
        print('Dialogue {vector} distance: {distance} '.format(vector=v, distance=distance))
    minDist = max(distances,key= lambda x: x[1])
    print(44*'_','\n','Smallest distance {small} belongs to the dialogue {vector} vector'.format(small=minDist[1],vector=minDist[0]))
    plt.scatter(vDict[minDist[0]],minDist[2])
    plt.show()
# Outputs the closest dialogue and session and the cosine distance in a tuple
def predict(dialogue_session, line):
    lowest = ('x',1)
    data = dataDict[dialogue_session][1][line,:]
    for vector in vDict:
        predictor = vDict[vector]
        if pair.pairwise_distances(predictor,data,'cosine') < lowest[1]:
            lowest = (vector, pair.pairwise_distances(predictor,data,'cosine'))
    return lowest

def predictV2(dialogue):
    pass


if __name__ == '__main__':
    # pickleArrays()
    # pickleVectors()
    dataDict, vDict = loadFiles()
    multiCompareV2('1')