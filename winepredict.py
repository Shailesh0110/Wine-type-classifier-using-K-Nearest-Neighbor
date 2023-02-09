import numpy as np
import  pandas as pd
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def main():

    df=pd.read_csv("WinePredictor (1).csv")

    #print(df.head())
    #print(df.columns)
    '''['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
           'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
           'Proanthocyanins', 'Color intensity', 'Hue',
           'OD280/OD315 of diluted wines', 'Proline'],'''
    # Creating labelEncoder
    label_encoder = preprocessing.LabelEncoder()
    df['Class']=label_encoder.fit_transform(df['Class'])

    df['Alcohol'] = label_encoder.fit_transform(df['Alcohol'])
    df['Malic acid'] = label_encoder.fit_transform(df['Malic acid'])
    df['Ash'] = label_encoder.fit_transform(df['Ash'])
    df['Alcalinity of ash'] = label_encoder.fit_transform(df['Alcalinity of ash'])
    df['Total phenols'] = label_encoder.fit_transform(df['Total phenols'])
    df['Flavanoids'] = label_encoder.fit_transform(df['Flavanoids'])
    df['Nonflavanoid phenols'] = label_encoder.fit_transform(df['Nonflavanoid phenols'])
    df['Proanthocyanins'] = label_encoder.fit_transform(df['Proanthocyanins'])
    df['Color intensity'] = label_encoder.fit_transform(df['Alcohol'])
    df['Alcohol'] = label_encoder.fit_transform(df['Color intensity'])
    df['Hue'] = label_encoder.fit_transform(df['Hue'])
    df['OD280/OD315 of diluted wines'] = label_encoder.fit_transform(df['OD280/OD315 of diluted wines'])
    df['Proline'] = label_encoder.fit_transform(df['Proline'])


    print(df)

    X=df.drop(columns=['Class'])
    #df['Class'] = label_encoder.fit_transform(df['Class'])
    print(X)
    #print(y)
    y=df['Class']

    X_train,X_test,y_train,y_test=train_test_split(X,y ,test_size=0.3)
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)

    accuracy=accuracy_score(y_test,y_pred)
    print("Accuracy:",accuracy * 100,"%")

if __name__=="__main__":
    main()

