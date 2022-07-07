import matplotlib as plt
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from Classes.Models.AbstractPredictionModel import PredictionModel

class NaiveBayes(PredictionModel):
    model = None
    def __init__(self):        
        self.model = MultinomialNB()
        self.name = "Naive Bayes"
