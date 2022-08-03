class Combinations():
    __ModelsHyperparameters = []
    __PerformanceMeasurements    = []
    
    
    def GetList_Hyperparameters(self):
        list = []
        for element in self.__ModelsHyperparameters[0].keys():
            list.append(element)
        return list
    
    def GetList_PerformanceMeasurements(self):
        list = []
        for element in self.__PerformanceMeasurements[0].keys():
            list.append(element)
        return list

    def Set_ModelsHyperparameters(self, hyperparameters):
        self.__ModelsHyperparameters = hyperparameters
    
    def SetSingle_PerformanceMeasurement(self, measurement):
        self.__PerformanceMeasurements.append(measurement)

    def Get_ModelsHyperparameters(self):
        return self.__ModelsHyperparameters

    def Get_PerformanceMeasurements(self):
        return self.__PerformanceMeasurements
    
    def GetSingle_Hyperparameter(self, index):
        return self.__ModelsHyperparameters[index]

    def GetSingle_PerformanceMeasurement(self, index):
        return self.__PerformanceMeasurements[index]
    
    def GetSingle_HyperparameterValues(self, index):
        return list(self.__ModelsHyperparameters[index].values())
        
    def GetSingle_PerformanceMeasurementValues(self, index):
        return list(self.__PerformanceMeasurements[index].values())

    def Count(self):
        return len(self.__ModelsHyperparameters)