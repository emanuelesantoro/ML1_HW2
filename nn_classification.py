import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import ParameterGrid

def reduce_dimension(features, n_components):
    """
    :param features: Data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality. Shape: (n_samples, n_components)
    """
    # Set the parameters
    pca = PCA(n_components=n_components,random_state=1)
    pca.fit(features)
    X_reduced = pca.transform(features)
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f'Explained variance: {explained_var}')
    return X_reduced

def train_nn(features, targets):
    """
    Train MLPClassifier with different number of neurons in one hidden layer.

    :param features:input data
    :param targets: corresponding label of input data
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, 
                                                        test_size=0.2, random_state=33)

    n_hidden_neurons = [2,10,100,200]
    # Set the parameters (some of them are specified in the HW2 sheet).
    losses=[]
    for n_hid in n_hidden_neurons:
        mlp = MLPClassifier(hidden_layer_sizes=(n_hid,), max_iter=500, 
                            solver='adam', random_state=1)
        mlp.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, mlp.predict(X_train))
        test_acc = accuracy_score(y_test, mlp.predict(X_test))
        loss= mlp.loss_
        print(f'N_hid: {n_hid}. Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')
        losses.append(mlp.loss_curve_)
    
    #Plot the loss curves
    for i, n_hid in enumerate(n_hidden_neurons):
        plt.plot(losses[i], label=f'n_hid={n_hid}')
    plt.xlabel('Number of Iteration')
    plt.ylabel('Log Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.show()
    

def train_nn_with_regularization(features, targets):
    """
    Train MLPClassifier using regularization.

    :param features:input data
    :param targets: corresponding label of input data
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    n_hidden_neurons = [2,10,100,200]

    print('alpha=0.1')
    for n_hid in n_hidden_neurons:
        mlp = MLPClassifier(hidden_layer_sizes=(n_hid,), max_iter=500, 
                            solver='adam', random_state=1, alpha=0.1)
        mlp.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, mlp.predict(X_train))
        test_acc = accuracy_score(y_test, mlp.predict(X_test))
        loss= mlp.loss_
        print(f'N_hid: {n_hid}. Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')

    print('early_stopping=True')
    for n_hid in n_hidden_neurons:
        mlp = MLPClassifier(hidden_layer_sizes=(n_hid,), max_iter=500, 
                            solver='adam', random_state=1,early_stopping=True)
        mlp.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, mlp.predict(X_train))
        test_acc = accuracy_score(y_test, mlp.predict(X_test))
        loss= mlp.loss_
        print(f'N_hid: {n_hid}. Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')

    
    print('alpha=0.1, early_stopping=True')
    for n_hid in n_hidden_neurons:
        mlp = MLPClassifier(hidden_layer_sizes=(n_hid,), max_iter=500, 
                            solver='adam', random_state=1,alpha=0.1, early_stopping=True)
        mlp.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, mlp.predict(X_train))
        test_acc = accuracy_score(y_test, mlp.predict(X_test))
        loss= mlp.loss_
        print(f'N_hid: {n_hid}. Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')

def train_nn_with_different_seeds(features, targets):
    """
    Train MLPClassifier using different seeds.
    Print (mean +/- std) accuracy on the training and test set.
    Print confusion matrix and classification report.

    :param features:input data
    :param targets: corresponding label of input data
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    seeds = [7,23,69,207,621] 

    train_acc_arr = []
    test_acc_arr = []
    losses=[]
    n_hid=200
    test_acc_arr = []
    for s in seeds:
        mlp = MLPClassifier(hidden_layer_sizes=(n_hid,), max_iter=500, 
                            solver='adam', random_state=s, alpha=0.1)
        mlp.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, mlp.predict(X_train))
        test_acc = accuracy_score(y_test, mlp.predict(X_test))
        test_acc_arr.append(test_acc)
        train_acc_arr.append(train_acc)
        losses.append(mlp.loss_curve_)
    
    test_acc_mean = np.mean(test_acc_arr)
    test_acc_std = np.std(test_acc_arr)
    test_acc_min = np.min(test_acc_arr)
    test_acc_max = np.max(test_acc_arr)

    train_acc_mean = np.mean(train_acc_arr)
    train_acc_std = np.std(train_acc_arr)
    train_acc_min = np.min(train_acc_arr)
    train_acc_max = np.max(train_acc_arr)
    
    print(f'Average accuracy on the train set: {train_acc_mean:.4f} +/- {train_acc_std:.4f}')
    print(f'Average accuracy the test set: {test_acc_mean:.4f} +/- {test_acc_std:.4f}')
    
    print(f'Minimum accuracy on the train set: {train_acc_min:.4f}, test set: {test_acc_min}' )
    print(f'Maximum accuracy on the train set: {train_acc_max:.4f}, test set: {test_acc_max}' )
    
    # #Plot the loss curves
    for i, s in enumerate(seeds):
        plt.plot(losses[i], label=f'seed value={s}')
    plt.xlabel('Number of Iteration')
    plt.ylabel('Log Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.show()
    
    print("Predicting on the test set")
    #Select the MLP classifier with the first seed
    pred_mlp = MLPClassifier(hidden_layer_sizes=(n_hid,), max_iter=500, solver='adam', random_state=seeds[0], alpha=0.1)
    pred_mlp.fit(X_train, y_train)
    
    #Make predictions on the test set using the best MLP classifier
    y_pred = pred_mlp.predict(X_test)
    
    # Print the classification report and confusion matrix
    print("Classification Report: \n",classification_report(y_test, y_pred))
    cm=confusion_matrix(y_test, y_pred, labels=range(10))
    print("Confusion Matix: \n",cm)
    tpArr=[0] *10
    fpArr=[0] *10
    fnArr=[0] *10
    for i in range(10):
        tpArr[i] = cm[i][i]
        fpArr[i] = sum(cm[j][i] for j in range(10)) - tpArr[i]
        fnArr[i] = sum(cm[i][j] for j in range(10)) - tpArr[i]
    error=np.array(fpArr)+np.array(fnArr)
    print(f'The most misclassified image is image {np.argmax(error)}.')
    #for i in range(10):
        #print(f'Recall of class {i} is {tpArr[i]/(tpArr[i]+fnArr[i])}')
        
    

def perform_grid_search(features, targets):
    """
    BONUS task: Perform GridSearch using GridSearchCV.
    Create a dictionary of parameters, then a MLPClassifier (e.g., nn, set default values as specified in the HW2 sheet).
    Create an instance of GridSearchCV with parameters nn and dict.
    Print the best score and the best parameter set.

    :param features: input data
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    
    #Create Dictionary
    parameters = {
        "alpha": [0.0, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "adam"],
        "activation": ["logistic", "relu"],
        "hidden_layer_sizes": [(100,), (200,)]
    }
    print(f'Number of model architectures to be tested: {len(ParameterGrid(parameters))}')
    nn = MLPClassifier(max_iter=100, random_state=1, learning_rate_init=0.01)
    grid_search = GridSearchCV(estimator=nn, param_grid=parameters, n_jobs=-1, verbose=False)
    grid_search.fit(X_train, y_train)
    # print message when training is complete
    print("\nTraining complete!")
    print(f'Best score: {grid_search.best_score_}')
    print(f'Best parameters: {grid_search.best_params_}')

    best_nn = grid_search.best_estimator_
    test_accuracy=accuracy_score(y_test,best_nn.predict(X_test))
    print(f'Test accuracy: {test_accuracy}')
