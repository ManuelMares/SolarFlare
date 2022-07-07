import numpy as np

from Classes.Models.SVM import SVM
from Classes.Models.RandomForest import RandomForest
from Classes.Models.KNN import KNN
from Classes.Models.DecisionTree import DecisionTree
from Classes.Models.NaiveBayes import NaiveBayes


class ModelsController:
    _SVM = SVM()
    _RandomForest = RandomForest()
    _KNN = KNN()
    _DecisionTree = DecisionTree()
    #_NaiveBayes = NaiveBayes()

    def Fit_Models(self, X_train, y_train):
        self._SVM.Fit_Model(X_train, y_train)
        self._RandomForest.Fit_Model(X_train, y_train)
        self._KNN.Fit_Model(X_train, y_train)
        self._DecisionTree.Fit_Model(X_train, y_train)
        #self.#_NaiveBayes.FastLoad(X_train, y_train)


    def Predict_Models(self, X_test):
        self._SVM.Predict(X_test)
        self._RandomForest.Predict(X_test)
        self._KNN.Predict(X_test)
        self._DecisionTree.Predict(X_test)
        #self.#_NaiveBayes.Predict(X_test)

    def Evaluate_Models(self, y_test, X_test = None):

        self._SVM.Evaluate_Model(y_test, X_test)
        self._RandomForest.Evaluate_Model(y_test, X_test)
        self._KNN.Evaluate_Model(y_test, X_test)
        self._DecisionTree.Evaluate_Model(y_test, X_test)
        #self.#_NaiveBayes.Evaluate_Model(y_test, X_test)


    def Print_Models(self):
        print(self._SVM)
        print(self._RandomForest)
        print(self._KNN)
        print(self._DecisionTree)
        #print(self.#_NaiveBayes)

        
    def Fit_SingleModel(name):
        try:
            return globals()[name]
        except:
            print(f"Model not identified while fitting model {name}")
    
    def Predict_SingleModel(self, name, X_test):
        try:
            globals()[name].Predict(X_test)
        except:
            print(f"Model not identified while predicting {name}")

    def Evaluate_SingleModel(self, name, y_test, X_test=None):
        try:
            globals()[name].Evaluate_Model(y_test, X_test)
        except:
            print(f"model not identified while evaluating {name}")
    
    def Print_SingleModel(self, name):
        try:
            print(name)
            a=eval(name)
            print(eval(name))
            print(locals()[name])
            print(globals()[name])
        except:
            print(f"model not indentified while printing {name}")

