import pandas as pd
import numpy as np
import pickle
import scipy.stats as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Classes.ModelsController import ModelsController

modelsController = ModelsController()

testSize = 0.1
hiddenDim = 128
numEpochs = 1





def Get_DataSet():
    labelFile  = "Resources/Sampled_labels.pck" 
    inputsFile = "Resources/Sampled_inputs.pck"
    
    sampledInputs = load_File(inputsFile)
    inputs_Scaled = Standardize_Data(sampledInputs)
    sampledLabels = load_File(labelFile)
    
    return Get_SummarizationMatrix(inputs_Scaled), Clasify_Labels(sampledLabels)

def load_File(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def Standardize_Data(data):  
    sc = StandardScaler()
    stdData=[]
    size = data.size
    for matrix in data:
        stdData.append( sc.fit_transform(matrix) )
    return np.array(stdData)

def Clasify_Labels(labels):
    labels[ labels == 0 ] = 0
    labels[ labels == 1 ] = 0
    labels[ labels == 2 ] = 1
    labels[ labels == 3 ] = 1
    return labels

def Get_SummarizationMatrix(inputs_Scaled):
    summarizationMatrix  = [ ]
    for matrix in inputs_Scaled:
        summarizationMatrix.append( Get_SummarizationArray(matrix) )
    #from list to array
    return np.array(summarizationMatrix)

def Get_SummarizationArray(matrix):   
    #returns the sumarization of each array by obtaining the sum of each row
    summarizationArray = []
    for row in matrix:            
        tsArray = Populate_tsArray( len(matrix[0])-1, row )
        summarizationRow = []

        summarizationRow.append( np.mean(row) )
        summarizationRow.append( np.std(row) )
        summarizationRow.append( st.skew(row) )
        summarizationRow.append( st.kurtosis(row) )
        summarizationRow.append( np.mean(tsArray) )
        summarizationRow.append( np.std(tsArray) )
        summarizationRow.append( st.skew(tsArray) )
        summarizationRow.append( st.kurtosis(tsArray)  )

        summarizationArray.append( summarizationRow )
    
    #returns a 1-dimensional list
    return np.concatenate(summarizationArray)

def Populate_tsArray(size, row):
    tsArray = np.zeros( size ) 
    for index in range (size):
        tsArray[index] = row[index + 1] - row[index]
    return tsArray    

def Set_Models(X_train, X_test, y_train, y_test):
    modelsController.Fit_Models(X_train, y_train)
    modelsController.Predict_Models(X_test)
    modelsController.Evaluate_Models(y_test)
    modelsController.Print_Models()

def ExportData_ToPCK(X, y ):
    pd.DataFrame(X).to_pickle("Resources/SolarFlare_Features.pck")
    pd.DataFrame(y).to_pickle("Resources/SolarFlare_Labels.pck")

def Get_ReadyData():
    inputsFile  = "Resources/SolarFlare_Features.pck" 
    labelFile = "Resources/SolarFlare_Labels.pck"
    
    return load_File(inputsFile), load_File(labelFile)


if __name__ == '__main__':
    #X, y = Get_DataSet()
    #Export_Data(X, y)    
    X, y = Get_ReadyData()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0, shuffle = True)
    Set_Models(X_train, X_test, y_train, y_test)


