import matplotlib as plt
import numpy as np
from sklearn import svm
from Classes.Models.AbstractPredictionModel import PredictionModel

class SVM(PredictionModel):
    model = None
    
    def __init__(self):        
        self.model = svm.SVC()
        self.name = "SVM"