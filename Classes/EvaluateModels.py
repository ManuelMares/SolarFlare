from sklearn.metrics import accuracy_score, confusion_matrix

def Get_ModelPerformanceMeassurements(y_test, y_predicted):
    modelMeasurements = {"accuracy": None, 
                        "positivePrecision": None, "negativePrecision": None, 
                        "positiveRecall": None, "negativeRecall": None, 
                        "positiveF1": None, "negativeF1": None,
                        "HSS1": None, "HSS2": None, 
                        "GS": None, "TSS": None}
    confusionMatrix = confusion_matrix(y_test, y_predicted)
    tP, fP, fN, tN = confusionMatrix[0][0], confusionMatrix[0][1], confusionMatrix[1][0], confusionMatrix[1][1]
    p, n = (confusionMatrix[0][0] + confusionMatrix[0][1]), (confusionMatrix[1][0] + confusionMatrix[1][1])
    p_, n_ = (confusionMatrix[0][0] + confusionMatrix[1][0]), (confusionMatrix[0][1] + confusionMatrix[1][1])

    modelMeasurements["accuracy"] = accuracy_score(y_test, y_predicted)
    modelMeasurements["positivePrecision"] = tP / (tP + fP)
    modelMeasurements["negativePrecision"] = tN / (tN + fN)
    modelMeasurements["positiveRecall"] = tP / (tP + fN)
    modelMeasurements["negativeRecall"] = tN / (tN + fP)
    modelMeasurements["positiveF1"] = ( 2 * modelMeasurements["positiveRecall"] * modelMeasurements["positivePrecision"]) / ( modelMeasurements["positiveRecall"] + modelMeasurements["positivePrecision"] )
    modelMeasurements["negativeF1"] = ( 2 * modelMeasurements["negativeRecall"] * modelMeasurements["negativePrecision"]) / ( modelMeasurements["negativeRecall"] + modelMeasurements["negativePrecision"] )
    modelMeasurements["HSS1"] = (tP + tN - n) / p
    modelMeasurements["HSS2"] = (2 * (tP*tN - fN*fP)) / ((p*n_) + (n*p_))
    modelMeasurements["GS"] = ( tP*(p+n) - (p*p_) ) / (fN*(p + n) - (n *p_))
    modelMeasurements["TSS"] = (tP - n) - (fP - n)

    return modelMeasurements
