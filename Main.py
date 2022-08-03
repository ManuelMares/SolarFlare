from Classes.ModelsManager import ModelsManager
from Classes.DatasetManager import Load_DataSet
from sklearn.model_selection import train_test_split
import Classes.ModelsEvaluator as ModelsEvaluator
from Classes.Enums.ModelsML import ModelsML
import csv


modelsManager = ModelsManager()
testSize = 0.1


def CreateCSV(modelsManager):
    combinations = modelsManager.GetSingle_Model(ModelsML.SVC).Combinations_Get_Combinations()
    
    

    with open('Resources/SVM_1.csv', 'w') as f:
        
        writer = csv.writer(f)
        
        header  = combinations.GetList_Hyperparameters() 
        header += combinations.GetList_PerformanceMeasurements()
        writer.writerow(header)

        for index in range(combinations.Count()):
            row  = combinations.GetSingle_HyperparameterValues(index)
            row += combinations.GetSingle_PerformanceMeasurementValues(index)
            writer.writerow(row)

        


if __name__ == '__main__':
    X, y = Load_DataSet()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0, shuffle = True)
    
    modelsManager.Create_Models()
    modelsManager.Test_Models(X_train, y_train, X_test, y_test)
    CreateCSV(modelsManager)
    





