import numpy as np


from Classes.Models.KNN import KNN
from Classes.Models.DecisionTree import DecisionTree
from Classes.Models.NB_Gaussian import NB_Gaussian
from Classes.Models.SVC import SVC
from Classes.Models.RandomForest import RandomForest

from Classes.Enums.ModelsML import ModelsML

class ModelsManager:
    models = {}

    # To operate all models at once -----------------------9
    def Create_Models(self):
        for modelEnum in ModelsML:
            #globals()[modelEnum.name] gets the string from enum as variable, '()' is to evaluate, is to create a instance
            self.models[modelEnum.value] = globals()[modelEnum.value]()

    def Test_Models(self, X_train, y_train, X_test, y_test):
        for model in self.models.values():
            count = model.Combinations_Count()
            for parametersIndex in range( count ):
                #This line set the new combination of hyperparameters
                model.ModelOperation_UpdateParameters(parametersIndex) 

                model.ModelOperation_Fit(X_train, y_train)
                model.ModelOperation_Predict(X_test)
                model.ModelOperation_Evaluate(y_test, X_test)

    def Fit_Models(self, X_train, y_train):
        for model in self.models.values():      
            #print(model)
            #print(X_train.describe(), y_train.describe()) 
            model.ModelOperation_Fit(X_train, y_train)

    def Predict_Models(self, X_test):
        for model in self.models.values():
            model.ModelOperation_Predict(X_test)

    def Evaluate_Models(self, y_test, X_test = None):
        for model in self.models.values():
            model.ModelOperation_Evaluate(y_test, X_test)

    def Print_Models(self):
        for model in self.models.values():
            print(model)



    # For single models -----------------------------------
    def CreateSingle_Model(self, modelName):
        try:
            self.models[modelName.name] = globals()[modelName.name]()
        except:
            print("Error: The given modelName is not an Enum type")

    def FitSingle_Model(self, X_train, y_train, modelName):
        try:
            self.models[modelName.name].ModelOperation_Fit(X_train, y_train)
        except:
            print("Error: The given modelName is not an Enum type")

    def PredictSingle_Model(self, X_test, modelName):
        try:
            self.models[modelName.name].ModelOperation_Predict(X_test)
        except:
            print("Error: The given modelName is not an Enum type")
            
    def EvaluateSingle_Model(self, modelName, y_test, X_test = None):
        try:
            self.models[modelName.name].ModelOperation_Evaluate(y_test, X_test)
        except:
            print("Error: The given modelName is not an Enum type")

    def PrintSingle_Model(self, modelName, y_test, X_test = None):
        try:            
            print(self.models[modelName.name])
        except:
            print("Error: The given modelName is not an Enum type")
    
    def GetSingle_Model(self, modelName):
        try:            
            return self.models[modelName.name]
        except:
            print("Error: The given modelName is not an Enum type")


