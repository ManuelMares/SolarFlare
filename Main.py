from Classes.ModelsManager import ModelsManager
from Classes.DatasetManager import Load_DataSet
from sklearn.model_selection import train_test_split


testSize = 0.1


def Set_Models(X_train, X_test, y_train, y_test):
    modelsManager.Fit_Models(X_train, y_train)
    modelsManager.Predict_Models(X_test)
    modelsManager.Evaluate_Models(y_test)


if __name__ == '__main__':
    X, y = Load_DataSet()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0, shuffle = True)
    
    
    #print(f"Data types --------------\nX_train: {type(X_train)}\nX_test: {type(X_test)}\ny_train: {type(y_train)}\ny_test: {type(y_test)}")
    modelsManager = ModelsManager()
    modelsManager.Create_Models()
    modelsManager.Fit_Models(X_train, y_train)
    modelsManager.Predict_Models(X_test = X_test)
    modelsManager.Evaluate_Models(y_test = y_test)
    modelsManager.Print_Models()



