import pickle
import numpy as np
import pandas as pd
from Classes.Enums.DatasetPath import DatasetPath
from sklearn.preprocessing import StandardScaler
import scipy.stats as st



def Process_NewDataSet(inputsDir, labelsDir):
    sampledInputs = _Load_FilePCK(inputsDir)
    inputs_Scaled = _Standardize_Data(sampledInputs)
    sampledLabels = _Load_FilePCK(labelsDir)
    X = _Get_SummarizationMatrix(inputs_Scaled)
    y = _Clasify_Labels(sampledLabels)
    _ExportData_ToPCK(X, y)
    
    return X, y 

def Load_DataSet(inputsDir = DatasetPath.InputsReady, labelsDir = DatasetPath.LabelsReady):
    #if given specific files to load, it will do it
    #if not given parameters, it will try to load the ready ones
    #if there are not ready ones, or there is any other problem, it will load and process the very original ones
    try:
        #I am not sure if when returning directly, having sent the first one and failing in the second one; there will be 3 return variables
        #that is why I save them in variables first
        print(f"------------Files to use -----------\n{inputsDir.value} \n{labelsDir.value}")
        inputs =_Load_FilePCK(inputsDir.value)
        labels = _Load_FilePCK(labelsDir.value).values.ravel()
        #print(f"returned data \ninputs: {type(inputs)}\nlabels: {type(labels)}")
        return inputs, labels
    except:
        sampledInputs = _Load_FilePCK(DatasetPath.Inputs.value)
        inputs_Scaled = _Standardize_Data(sampledInputs)
        sampledLabels = _Load_FilePCK(DatasetPath.Labels.value)
        inputs = _Get_SummarizationMatrix(inputs_Scaled)
        labels = _Clasify_Labels(sampledLabels).values.ravel()

        #print(f"returned data \ninputs: {type(inputs)}\nlabels: {type(labels)}")
        _ExportData_ToPCK(inputs, labels)
        
        return inputs, labels


# Private methods -----------------------------------------------------------------------------------------------

def _Load_FilePCK(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def _Standardize_Data(data):  
    sc = StandardScaler()
    stdData=[]
    for matrix in data:
        stdData.append( sc.fit_transform(matrix) )
    return np.array(stdData)

def _Clasify_Labels(labels):
    labels[ labels == 0 ] = 0
    labels[ labels == 1 ] = 0
    labels[ labels == 2 ] = 1
    labels[ labels == 3 ] = 1
    return labels

def _Get_SummarizationMatrix(inputs_Scaled):
    summarizationMatrix  = [ ]
    for matrix in inputs_Scaled:
        summarizationMatrix.append(_Get_SummarizationArray(matrix) )
    #from list to array
    return np.array(summarizationMatrix)

def _Get_SummarizationArray(matrix):   
    #returns the sumarization of each array by obtaining the sum of each row
    summarizationArray = []
    for row in matrix:            
        tsArray = _Populate_tsArray( len(matrix[0])-1, row )
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

def _Populate_tsArray(size, row):
    tsArray = np.zeros( size ) 
    for index in range (size):
        tsArray[index] = row[index + 1] - row[index]
    return tsArray      

def _ExportData_ToPCK(X, y ):
        from Classes.Enums.DatasetPath import Add_File

        fileIndex = len(DatasetPath) / 2
        InputsName = f"Inputs_{fileIndex}"
        LabelsName = f"Labels_{fileIndex}"

        pd.DataFrame(X).to_pickle(f"Resources/{InputsName}.pck")
        pd.DataFrame(y).to_pickle(f"Resources/{LabelsName}.pck")
        Add_File(f"{InputsName}", f"Resources/{InputsName}.pck")
        Add_File(f"{LabelsName}", f"Resources/{LabelsName}.pck")

