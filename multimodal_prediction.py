from sklearn import decomposition as dc
import numpy as np
from sklearn.metrics import mean_squared_error as mse

modalitiesArray = np.loadtxt('4m1.data')
valueNamesArray = np.loadtxt('4m1.head', dtype=bytes)

def b2s(b):
    return str(b,'utf-8')
b2sV = np.vectorize(b2s)
valueNamesArray = b2sV(valueNamesArray)

print(modalitiesArray.shape)
print(valueNamesArray.shape)
# Copy used to average the unchanged values
modalitiesArraycp = np.copy(modalitiesArray)
for i in list(range(modalitiesArray.shape[0])):
    for j in list(range(modalitiesArray.shape[1])):
        if np.isnan(modalitiesArray[i,j]):
            column = modalitiesArraycp[:,j]
            colsum = column[~np.isnan(column)].sum()
            colavg = colsum/np.count_nonzero(~np.isnan(column))
            modalitiesArray[i,j] = colsum


nnmf = dc.NMF(n_components=60)
W = nnmf.fit_transform(modalitiesArray)
H = nnmf.components_
predictedMod = np.dot(W, H)

# errorList = [predictedMod[:,x].mean()-modalitiesArray[:,x].mean() for x in range(modalitiesArray.shape[1])]

#Testing

testArray = modalitiesArray
testArray[-1,-21:-1] = testArray[-1,0:20].mean()

# nnmfTest = dc.NMF(n_components=60)
# Wtest = nnmfTest.fit_transform(testArray)
# Htest = nnmfTest.components_
# predictedT = np.dot(Wtest,Htest)

# normalizedRMSE = np.sqrt(mse(modalitiesArray[-1,-21:-1],predictedT[-1,-21:-1]))/(np.concatenate((
#     modalitiesArray[-1,-21:-1],predictedT[-1,-21:-1])).max() - np.concatenate((
#     modalitiesArray[-1,-21:-1],predictedT[-1,-21:-1])).min())

def normalizedRMSE(real,predicted):

    return np.sqrt(mse(real,predicted))/(np.concatenate((real,predicted)).max() - np.concatenate((real,predicted)).min())


def normalizedByMeanRMSE(real,predicted):

    return np.sqrt(mse(real,predicted))/(np.concatenate((real,predicted)).mean())


# print('Test:',normalizedRMSE(modalitiesArray[-1,-21:-1],predictedT[-1,-21:-1]))
# print('Original', normalizedRMSE(modalitiesArray[-1,-21:-1],predictedMod[-1,-21:-1]))
#
# print('Test(mean norm):',normalizedByMeanRMSE(modalitiesArray[-1,-21:-1],predictedT[-1,-21:-1]))
# print('Original(mean norm)', normalizedByMeanRMSE(modalitiesArray[-1,-21:-1],predictedMod[-1,-21:-1]))

#Chooses the column to test, and the last given amount of values
def errorComp(column, amount=20):
    testArray = np.copy(modalitiesArray)
    if testArray[0:-amount-2,column].mean() == 0:
        return print('zero error')
    testArray[-amount-1:-1,column] = testArray[0:-amount-2,column].mean()

    nnmfTest = dc.NMF(n_components=1)
    Wtest = nnmfTest.fit_transform(testArray)
    Htest = nnmfTest.components_
    predictedT = np.dot(Wtest,Htest)
    # Prints out the normalized RMSE for the test array
    print('Test:',normalizedRMSE(modalitiesArray[-amount-1:-1,column],predictedT[-amount-1:-1,column]))
    # Prints out the normalized RMSE for the original
    print('Original', normalizedRMSE(modalitiesArray[-amount-1:-1,column], predictedMod[-amount - 1:-1, column]))
    # Uses the mean to normalize instead of the range
    print('Test(mean norm):',normalizedByMeanRMSE(modalitiesArray[-amount-1:-1,column],predictedT[-amount-1:-1,column]))
    print('Original(mean norm)', normalizedByMeanRMSE(modalitiesArray[-amount-1:-1,column], predictedMod[-amount - 1:-1, column]))

if __name__ == '__main__':

    errorComp(5,200)










