import numpy as np
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
import csv
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#Extracting data from csv file into X input set and Y output set for the 3 classes
with open('E:\Semester10\Machine Learning\Classification Project\clean_2014.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    n_samples = 141246
    n_features = 59
    Y = np.zeros(n_samples)
    X = np.zeros((n_samples,n_features))
    i = 0
    for line in csv_reader:
         j = 0
         Y[i] = int(line[11])
         for k in range(len(line)):
             if (k != 0):
                 if(k != 1):
                     if(k != 11):
                        X[i][j] = float(line[k])
                        j += 1        
         i += 1
n_classes = 3
       
        
#For Stratified Sampling, First Step is to stratify the data by class
def StratifyByClass(X,Y,n_classes):
    #Classes Count
    zerosCount = 119903
    onesCount = 19792
    twosCount = 1551
    
    #Creating Classes Sets
    Xzeros = np.zeros((zerosCount,X.shape[1]))
    Yzeros = np.zeros((zerosCount,n_classes))
    
    Xones = np.zeros((onesCount,X.shape[1]))
    Yones = np.zeros((onesCount,n_classes))
    
    Xtwos = np.zeros((twosCount,X.shape[1]))
    Ytwos = np.zeros((twosCount,n_classes))
    
    #Using 1 hot encoding to encode the outputs 0,1 and 2 as we use Segmoid function
    zeros = 0
    ones = 0
    twos = 0
    for i in range(X.shape[0]):
        if (Y[i] == 0):
            Xzeros[zeros] = X[i]
            Yzeros[zeros] = [1,0,0]
            zeros += 1
            
        if (Y[i] == 1):
            Xones[ones] = X[i]
            Yones[ones] = [0,1,0]
            ones += 1
            
        if (Y[i] == 2):
            Xtwos[twos] = X[i]
            Ytwos[twos] = [0,0,1]
            twos += 1
    
    return(Xzeros,Yzeros,Xones,Yones,Xtwos,Ytwos)


#Second Step to randomly chose instances from each class proportionally and construct training and testing sets
def StratifiedSampling(X,Y):
    Xzeros,Yzeros,Xones,Yones,Xtwos,Ytwos = StratifyByClass(X,Y,3)
    
    # Get the total number of rows in the original array
    total_rows = Xzeros.shape[0]+Xones.shape[0]+Xtwos.shape[0]
    
    # Construction Train, CV, and Test sets
    zerosWeight =  Xzeros.shape[0]/total_rows
    onesWeight =  Xones.shape[0]/total_rows
    twosWeight =  Xtwos.shape[0]/total_rows
    
    train_size = int(0.6*total_rows)
    cv_size = int(0.2*total_rows)
    test_size = int(0.2*total_rows)
    
    train_zeros = int(zerosWeight*train_size)
    cv_zeros = int(zerosWeight*cv_size)
    test_zeros = int(zerosWeight*test_size)
    
    train_ones = int(onesWeight*train_size)
    cv_ones = int(onesWeight*cv_size)
    test_ones = int(onesWeight*test_size)
    
    train_twos = int(twosWeight*train_size)
    cv_twos = int(twosWeight*cv_size)
    test_twos = int(twosWeight*test_size)
    
    # Create an array of indices representing available rows
    zeros_available_indices = np.arange(Xzeros.shape[0])
    ones_available_indices = np.arange(Xones.shape[0])
    twos_available_indices = np.arange(Xtwos.shape[0])
    
    # Randomly select row indices without replacement
    zeros_selected_indices = np.random.choice(zeros_available_indices, train_zeros, replace=False)
    ones_selected_indices = np.random.choice(ones_available_indices, train_ones, replace=False)
    twos_selected_indices = np.random.choice(twos_available_indices, train_twos, replace=False)

    # Remove the selected indices from the available indices
    mask = np.isin(zeros_available_indices, zeros_selected_indices, invert=True)
    zeros_available_indices = zeros_available_indices[mask]
    mask = np.isin(ones_available_indices, ones_selected_indices, invert=True)
    ones_available_indices = ones_available_indices[mask]
    mask = np.isin(twos_available_indices, twos_selected_indices, invert=True)
    twos_available_indices = twos_available_indices[mask]
    
    # Create the new array by indexing the original array with the selected indices
    trainZ = Xzeros[zeros_selected_indices, :]
    trainO = Xones[ones_selected_indices, :]
    trainT = Xtwos[twos_selected_indices, :]
    trainZY = Yzeros[zeros_selected_indices, :]
    trainOY = Yones[ones_selected_indices, :]
    trainTY = Ytwos[twos_selected_indices, :]
    Xtrain = np.concatenate((trainZ, trainO, trainT), axis=0)
    Ytrain = np.concatenate((trainZY, trainOY, trainTY), axis=0)

    # Randomly select row indices without replacement
    zeros_selected_indices = np.random.choice(zeros_available_indices, cv_zeros, replace=False)
    ones_selected_indices = np.random.choice(ones_available_indices, cv_ones, replace=False)
    twos_selected_indices = np.random.choice(twos_available_indices, cv_twos, replace=False)
    
    # Remove the selected indices from the available indices
    mask = np.isin(zeros_available_indices, zeros_selected_indices, invert=True)
    zeros_available_indices = zeros_available_indices[mask]
    mask = np.isin(ones_available_indices, ones_selected_indices, invert=True)
    ones_available_indices = ones_available_indices[mask]
    mask = np.isin(twos_available_indices, twos_selected_indices, invert=True)
    twos_available_indices = twos_available_indices[mask]
    
    # Create the new array by indexing the original array with the selected indices
    cvZ = Xzeros[zeros_selected_indices, :]
    cvO = Xones[ones_selected_indices, :]
    cvT = Xtwos[twos_selected_indices, :]
    cvZY = Yzeros[zeros_selected_indices, :]
    cvOY = Yones[ones_selected_indices, :]
    cvTY = Ytwos[twos_selected_indices, :]
    Xcv = np.concatenate((cvZ, cvO, cvT), axis=0)
    Ycv = np.concatenate((cvZY, cvOY, cvTY), axis=0)
    
    # Randomly select row indices without replacement
    zeros_selected_indices = np.random.choice(zeros_available_indices, test_zeros, replace=False)
    ones_selected_indices = np.random.choice(ones_available_indices, test_ones, replace=False)
    twos_selected_indices = np.random.choice(twos_available_indices, test_twos, replace=False)
    
    # Remove the selected indices from the available indices
    mask = np.isin(zeros_available_indices, zeros_selected_indices, invert=True)
    zeros_available_indices = zeros_available_indices[mask]
    mask = np.isin(ones_available_indices, ones_selected_indices, invert=True)
    ones_available_indices = ones_available_indices[mask]
    mask = np.isin(twos_available_indices, twos_selected_indices, invert=True)
    twos_available_indices = twos_available_indices[mask]
    
    # Create the new array by indexing the original array with the selected indices
    testZ = Xzeros[zeros_selected_indices, :]
    testO = Xones[ones_selected_indices, :]
    testT = Xtwos[twos_selected_indices, :]
    testZY = Yzeros[zeros_selected_indices, :]
    testOY = Yones[ones_selected_indices, :]
    testTY = Ytwos[twos_selected_indices, :]
    Xtest = np.concatenate((testZ, testO, testT), axis=0)
    Ytest = np.concatenate((testZY, testOY, testTY), axis=0)
    
    return(Xtrain,Xcv,Xtest,Ytrain,Ycv,Ytest)
        

#Decoding when necessary for model evaluation simplicity 
def ClassDecode(y) :
    Y = np.zeros(len(y))
    for i in range(len(y)):
        if ((y[i]==[1,0,0]).all()):
            Y[i] = 0
        else:
           if ((y[i]==[0,1,0]).all()):
               Y[i] = 1
           else:
               if ((y[i]==[0,0,1]).all()):
                   Y[i] = 2
               else:
                   Y[i] = -1
    return(Y)

#Cross validation error
def Jcv(y_pred, y_cv):
    sqEr = 0
    for i in range(len(y_cv)):
        sqEr += (y_pred[i] - y_cv[i])**2
        
    error = (1/(2*len(y_cv)))*sqEr
    return (error)

#Model selection by sweeping on the hyperparameters alpha, lambda and the polynomial degree
#Using Cross Validation Set to avoid over fitting and leaving a testing set afterwards
def ModelSelection(d, lr, alpha, Xtrain,Xcv,Ytrain,Ycv):
    cvErr = np.zeros((len(d),len(lr),len(alpha)))
    Accuracy = np.zeros((len(d),len(lr),len(alpha)))
    for i in range(len(d)):
        for j in range(len(lr)):
            for k in range(len(alpha)):
                clf = LogisticRegression(d=d[i],lr=lr[j], alpha=alpha[k])
                clf.fit(Xtrain, Ytrain)
                y_pred = clf.predict(Xcv)
                Y_pred = ClassDecode(y_pred)
                Y_cv = ClassDecode(Ycv)
                cvErr[i][j][k] = Jcv(Y_pred, Y_cv)
                Accuracy[i][j][k] = accuracy(y_pred, Ycv)[0]
           
            
    ic,jc,kc = np.unravel_index(np.argmin(cvErr), cvErr.shape)
    icm,jcm,kcm = np.unravel_index(np.argmax(Accuracy), Accuracy.shape)
    
    return(d[ic], lr[jc], alpha[kc], cvErr, d[icm], lr[jcm], alpha[kcm], Accuracy)
  
#one way for model evaluation
def accuracy(y_pred, y_test):
    acc = 0 #for general correct predictions 
    TrueZeros = 0
    TrueOnes = 0
    TrueTwos = 0
    PredZeros = 0
    PredOnes = 0
    PredTwos = 0
    TestZeros = 0
    TestOnes = 0
    TestTwos = 0
    FalseZeros = 0
    FalseOnes = 0
    FalseTwos = 0
    Undefined = 0 # for undefined predictions 
    for i in range(len(y_pred)):
        if ((y_pred[i]==[1,0,0]).all()):
            PredZeros += 1
        else:
           if ((y_pred[i]==[0,1,0]).all()):
               PredOnes += 1
           else:
               if ((y_pred[i]==[0,0,1]).all()):
                   PredTwos += 1
               else:
                   Undefined += 1
                
        if ((y_test[i]==[1,0,0]).all()):
            TestZeros += 1
        if ((y_test[i]==[0,1,0]).all()):
            TestOnes += 1
        if ((y_test[i]==[0,0,1]).all()):
            TestTwos += 1
            
        if ((y_pred[i]==y_test[i]).all()):
            acc += 1
            if ((y_pred[i]==[1,0,0]).all()):
                TrueZeros += 1
            if ((y_pred[i]==[0,1,0]).all()):
                TrueOnes += 1
            if ((y_pred[i]==[0,0,1]).all()):
                TrueTwos += 1
                
    FalseZeros = PredZeros - TrueZeros
    FalseOnes = PredOnes - TrueOnes
    FalseTwos = PredTwos - TrueTwos      
      
    return (acc/(len(y_test)), PredZeros, PredOnes, PredTwos, TrueZeros, TrueOnes, TrueTwos, FalseZeros, FalseOnes, FalseTwos,TestZeros, TestOnes, TestTwos, Undefined)
      
#Confusion matrix for model evaluation
def plot_confusion_matrix(y_test,y_pred):
    cm = confusion_matrix(y_test,y_pred)
    sns.heatmap(cm,cmap='Blues',annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('original')
    plt.show()

def calculate_metrics(y_test,y_pred):
    # Calculate the metrics for this fold
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test,y_pred,average='weighted')
    return accuracy,precision,recall,f1

#Starting the model selection, testing and evaluation process

# Getting the necessary sets
Xtrain,Xcv,Xtest,Ytrain,Ycv,Ytest = StratifiedSampling(X,Y)

#learning rate values for model selection
lrS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3] 

#regularization parameter values for model selection
alphaS = [0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24]

#ploynomial degree values for model selection
dS = [1,2,3,4,5,6,7,8,9]

#Model Selection
dc, lrc, alphac, cvErr, dcm, lrcm, alphacm, Accuracy = ModelSelection(dS, lrS, alphaS, Xtrain,Xcv,Ytrain,Ycv)
#after training using training set, model selection using cross validation set
#the model hyperparameters are chosen
clf = LogisticRegression(d=1, lr=0.03, alpha=1.28)

clf.fit(Xtrain, Ytrain) #training
y_pred = clf.predict(Xtest) #testing
Y_pred = ClassDecode(y_pred)
Y_test = ClassDecode(Ytest)

# Evaluation

acc,PZ, PO, PT, TZ, TO, TT, FZ,FO,FT,TsZ,TsO,TsT,Undef = accuracy(y_pred, Ytest)

accuracy,precision,recall,f1 = calculate_metrics(Y_test,Y_pred)

plot_confusion_matrix(Y_test, Y_pred)

print(f'  Accuracy: {accuracy:.4f}')
print(f'  Precision: {precision:.4f}')
print(f'  Recall: {recall:.4f}')
print(f'  F1 Score: {f1:.4f}')

print("accuracy : ", acc)
print("Predicted Zero : ",PZ)
print("Predicted One : ",PO)
print("Predicted two : ",PT)
print("True Zero : ",TZ)
print("True One : ",TO)
print("True Two : ",TT)
print("False Zero : ",FZ)
print("False One : ",FO)
print("False Two : ",FT)
print("Actual Zeros : ",TsZ)
print("Actual Ones : ",TsO)
print("Actual Twos : ",TsT)
print("Undefined prediction : ",Undef)