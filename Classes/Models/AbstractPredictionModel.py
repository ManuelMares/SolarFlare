from abc import ABC, abstractmethod 
import Classes.EvaluateModels as EvaluateModels

class PredictionModel(ABC):

    model = None    
    meassurements = None
    y_pred = None
    name = ""

    @abstractmethod
    def __init__(self):
        pass
    
    def Fit_Model(self, X_train, y_train):             
        self.model.fit(X_train, y_train)

    def Predict(self, X_test):
        self.y_pred = self.model.predict(X_test)

        
    def Evaluate_Model(self, y_test, X_test = None):
        prediction = self.y_pred if X_test is None else self.model.predict(X_test)

        self.meassurements = EvaluateModels.Get_ModelPerformanceMeassurements(y_test, prediction)

    def __Get_Meassurements(self):
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
        if self.meassurements is None:
            obj += "This model has not been meassured yet\n"
        else:
            obj += self.__Get_Meassurements()

        return obj
        
    def FastLoad(self, X_train, X_test, y_train):
        self.Fit_Model(X_train, y_train)
        self.y_pred = self.Predict(X_test)
