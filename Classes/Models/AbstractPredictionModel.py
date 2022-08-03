from abc import ABC, abstractmethod

from torch import combinations 
import Classes.ModelsEvaluator as ModelsEvaluator
from Classes.Combinations import Combinations

class PredictionModel(ABC):

    #each class lists its own parameters
    #each class has its 

    model = None    
    _Combinations = Combinations()
    y_pred = None
    hyperparameters = None
    name = ""

    @abstractmethod
    def __init__(self):
        pass
    
    def ModelOperation_Fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def ModelOperation_Predict(self, X_test):
        self.y_pred = self.model.predict(X_test)

    def ModelOperation_Evaluate(self, y_test, X_test = None):
        prediction = self.y_pred if X_test is None else self.model.predict(X_test)

        measurement = ModelsEvaluator.Get_ModelPerformanceMeasurements(y_test, prediction)
        self._Combinations.SetSingle_PerformanceMeasurement( measurement )
        print(f"new Measurement register:\n{self.name}\n{measurement}")

    def ModelOperation_UpdateParameters(self, parametersIndex):
        pass


    #Combination related methods
    def Combinations_Set_HyperParameters(self, hyperparameters):
        self._Combinations.Set_ModelsHyperparameters( hyperparameters )
    
    def _Combinations_SetSingle_PerformanceMeasurement(self, measurement):
        self._Combinations.SetSingle_PerformanceMeasurement( measurement )

    def Combinations_Get_HyperParameters(self):
        return self._Combinations.Get_ModelsHyperparameters()
    
    def Combinations_Get_PerformanceMeasurements(self):
        return self._Combinations.Get_PerformanceMeasurements()
    
    def Combinations_GetSingle_PerformanceMeasurement(self, index):
        return self._Combinations.GetSingle_PerformanceMeasurement(index)

    def Combinations_Get_Combinations(self):
        return self._Combinations

    def Combinations_Count(self):
        return self._Combinations.Count()


    def __Get_StringMeassurements(self):
        obj = ""
        for key in self.meassurements: 
            value = self.meassurements[key]       
            if value is None:
                obj += f"parameter {key} has not been meassured.\n"
            else:
                obj += f"{key:<17} {value:>5.2f}\n"
        return obj

    def __str__(self):
        obj = f"========== {self.name} =========\n"
        #if self.meassurements is None:
        #    obj += "This model has not been meassured yet\n"
        #else:
        #    obj += self.__Get_StringMeassurements()
#
        return obj
