'''In this assignment students need to predict whether a person makes over 50K per year
or not from classic adult dataset using XGBoost. The description of the dataset is as
follows:

Data Set Information:
Extraction was done by Barry Becker from the 1994 Census database. A set of
reasonably clean records was extracted using the following conditions: ((AAGE>16) &&
(AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

Attribute Information:
Listing of attributes:
>50K, <=50K.
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov,
Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc,
9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.

education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial,
Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing,
Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany,
Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras,
Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France,
Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong,
Holand-Netherlands.
Following is the code to load required libraries and data:
import numpy as np
import pandas as pd
train_set =
pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.dat
a', header = None)

test_set =
pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
, skiprows = 1, header = None)
col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
'occupation','relationship', 'race', 'sex', capital_gain', 'capital_loss', 'hours_per_week',
'native_country', 'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels
NOTE:​ ​The​ ​solution​ ​shared​ ​through​ ​Github​ ​should​ ​contain​ ​the​ ​source​ ​code​ ​used​ ​and​'''



# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 21:28:46 2018

@author: hp-pc
"""

#a=train_set['wage_class'].astype('category')
#a.cat.codes
#dict( enumerate(a.cat.categories) )

import numpy as np
import pandas as pd
def Label_encoder(df):
    df_labels = df.copy()
    num_cols=df.select_dtypes(exclude = [np.number,np.int16,np.bool,np.float32] )
    for col in num_cols:
        df[col] = df[col].astype('category')
        df_labels[col] = df_labels[col].astype('category')
        df_labels[col] = df_labels[col].cat.codes
    return df_labels

def model_training(X_train,y_train):
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    import xgboost
    try:
         # save the model 1 to disk
        classifier_model = xgboost.XGBClassifier()
        classifier_model.fit(X_train,y_train)
        filename = 'classifier_model.sav'
        pickle.dump(classifier_model, open(filename, 'wb'))
        return filename
    except Exception as e:
        print("Error occurs during model training")

def prediction(model_path,X_test):
    import pickle
    model = pickle.load(open(model_path, 'rb'))
    y_pred= model.predict(X_test)
    return y_pred

def classifier_accuracy(y_test,y_pred):
    from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
     #Accuracy and confusion matrix
    cm= confusion_matrix(y_test,y_pred)
    acc=accuracy_score(y_test,y_pred)
    class_report= classification_report(y_test,y_pred)
    
    
    print('Confusion Matrix : {} '.format(cm))
    print('Accuracy : {} '.format(acc))
    print('Class Report : {}'.format(class_report))

def get_data():
    train_set =pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header = None)
    test_set =pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', skiprows = 1, header = None)
    #test_set =(pd.read_csv('Assignment10.1_test_data.csv', skiprows = 1, header = None))
    col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                  'marital_status','occupation','relationship', 'race', 'sex', 'capital_gain', 
                  'capital_loss', 'hours_per_week','native_country', 'wage_class']
    train_set.columns = col_labels
    test_set.columns = col_labels
    return train_set,test_set

def data_preprocessing(train_set,test_set):
    
    #train_set['wage_class'] = np.where(train_set['wage_class']=='>50k',1,0)
    label_train_set= Label_encoder(train_set)

    #test_set['wage_class'] = np.where(test_set['wage_class']=='>50k',1,0)
    label_test_set= Label_encoder(test_set)
    return label_train_set,label_test_set

def get_train_test_data(train_set,test_set):
    X_train = train_set.iloc[:,:-1].values
    y_train = train_set.iloc[:,-1].values
    
    X_test = test_set.iloc[:,:-1].values
    y_test = test_set.iloc[:,-1].values
    
    return X_train,y_train,X_test,y_test

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
    import pickle
    train_set,test_set=get_data()
    dp_train_set,dp_test_set=data_preprocessing(train_set,test_set)
    X_train,y_train,X_test,y_test=get_train_test_data(dp_train_set,dp_test_set)
    model_path = model_training(X_train,y_train)
    y_pred = prediction(model_path,X_test)
    classifier_accuracy(y_test,y_pred)
    

