import numpy as np


from Classes.Models.KNN import KNN
from Classes.Models.DecisionTree import DecisionTree
from Classes.Models.NaiveBayes import NaiveBayes
from Classes.Models.SVM import SVM
from Classes.Models.RandomForest import RandomForest

from Classes.Enums.ModelsML import ModelsML

class ModelsManager:
    models = {}
    

    # To operate all models at once -----------------------9
    def Create_Models(self):
        for modelEnum in ModelsML:
            #globals()[modelEnum.name] gets the string from enum as variable, '()' is to evaluate, is to create a instance
            self.models[modelEnum.value] = globals()[modelEnum.value]()

    def Fit_Models(self, X_train, y_train):
        for model in self.models.values():      
            #print(model)
            #print(X_train.describe(), y_train.describe()) 
            model.Fit_Model(X_train, y_train)

    def Predict_Models(self, X_test):
        for model in self.models.values():
            model.Predict(X_test)

    def Evaluate_Models(self, y_test, X_test = None):
        for model in self.models.values():
            model.Evaluate_Model(y_test, X_test)

    def Print_Models(self):
        for model in self.models.values():
            print(model)

    # For single models -----------------------------------
    def Create_SingleModel(self, modelName):
        try:
            self.models[modelName.name] = globals()[modelName.name]()
        except:
            print("Error: The given modelName is not an Enum type")
            print("-----")

    def Fit_SingleModel(self, X_train, y_train, modelName):
        try:
            self.models[modelName.name].Fit_Model(X_train, y_train)
        except:
            print("Error: The given modelName is not an Enum type")
            print("-----")

    def Predict_SingleModel(self, X_test, modelName):
        try:
            self.models[modelName.name].Predict(X_test)
        except:
            print("Error: The given modelName is not an Enum type")
            print("-----")
            
    def Evaluate_SingleModel(self, modelName, y_test, X_test = None):
        try:
            self.models[modelName.name].Evaluate_Model(y_test, X_test)
        except:
            print("Error: The given modelName is not an Enum type")
            print("-----")

    def Evaluate_SingleModel(self, modelName, y_test, X_test = None):
        try:            
            print(self.models[modelName.name])
        except:
            print("Error: The given modelName is not an Enum type")
            print("-----")


