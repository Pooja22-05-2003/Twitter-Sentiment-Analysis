from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

plt.style.use('dark_background')

def EvaluatingModel(model,X_test,y_test,font):
    
    # Predict values for Test dataset
    y_pred = model.predict(X_test) #Xtest is not used in model training
    acc = accuracy_score(y_test, y_pred)*100
    # Print the evaluation metrics for the dataset.
    cr = classification_report(y_test, y_pred)
    print(cr)
    
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos'] #configuration of a confusin matrix
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)] #converting confusion matrix value to percentage in 2 decimal places.

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot = labels, cmap = 'PiYG',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    plt.show()
    
    l = cr.split('\n')
    lines=[]
    for i in range(len(l)):
        if(len(l[i])!=0):
            lines.append(l[i])
    for i in range(len(lines)):
        lines[i] = lines[i].strip()
    ws=[]
    for line in lines:
        ws.append(line.split())
    df_f1 = pd.DataFrame({'f1 score': pd.Series([ws[1][3],ws[2][3]])})
    # df_f1['class']=df_f1['class'].astype('float')
    df_f1['f1 score']=df_f1['f1 score'].astype('float')
    
    return acc, df_f1

