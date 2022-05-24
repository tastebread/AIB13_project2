import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from category_encoders import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import  make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_validate, RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from tensorflow import keras


os.chdir('C:/test/spaceship-titanic')

space_data_tr = pd.read_csv('train.csv')
space_data_te = pd.read_csv('test.csv')
submission = pd.read_csv('../../../Downloads/sample_submission.csv')

space_data_tr.drop(['PassengerId', 'Name'], axis = 1, inplace = True)
space_data_te.drop(['PassengerId', 'Name'], axis = 1, inplace = True)

space_data_tr['Deck'] = space_data_tr['Cabin'].str.split('/').str[0]
space_data_tr['Num'] = space_data_tr['Cabin'].str.split('/').str[1]
space_data_tr['Side'] = space_data_tr['Cabin'].str.split('/').str[2]

space_data_te['Deck'] = space_data_te['Cabin'].str.split('/').str[0]
space_data_te['Num'] = space_data_te['Cabin'].str.split('/').str[1]
space_data_te['Side'] = space_data_te['Cabin'].str.split('/').str[2]

space_data_tr.drop(['Cabin'], axis = 1, inplace = True)
space_data_te.drop(['Cabin'], axis = 1, inplace = True)

def encodings(df):
    features = ['CryoSleep', 'VIP']
    for feature in features:
        mapping = {True:1, False:0, 'N':2}
        df[feature] = df[feature].map(mapping)

    return df

def Label_encodings(df):
    features = ['HomePlanet', 'Destination', 'Side']
    for feature in features:
        label = LabelEncoder()
        df[feature]= label.fit_transform(df[feature])
    return df

def fillna(df):
    df['CryoSleep'].fillna('N', inplace=True)
    df['Destination'].fillna('N', inplace=True)
    df['HomePlanet'].fillna('N', inplace=True)
    df['VIP'].fillna('N', inplace=True)
    df['RoomService'].fillna(0, inplace=True)
    df['FoodCourt'].fillna(0, inplace=True)
    df['ShoppingMall'].fillna(0, inplace=True)
    df['Spa'].fillna(0, inplace=True)
    df['VRDeck'].fillna(0, inplace=True)
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Num'].fillna(0, inplace=True)
    df['Side'].fillna('N', inplace=True)


    return df

def transforms(df):
    df = fillna(df)
    df = encodings(df)
    df = Label_encodings(df)


    return df

space_data_tr = pd.get_dummies(space_data_tr, columns = ['Deck'])
space_data_te = pd.get_dummies(space_data_te, columns = ['Deck'])

space_data_tr = transforms(space_data_tr)
space_data_te = transforms(space_data_te)

X_space_data = space_data_tr.drop(columns = ['Transported'])
y_space_data = space_data_tr['Transported']

X_train, X_test, y_train, y_test = train_test_split(X_space_data, y_space_data, test_size = 0.2, random_state = 0)

model = Sequential()

model.add(BatchNormalization(input_shape= (X_train.shape[1],)))

model.add(Dense(1024, input_shape= (X_train.shape[1],), activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1024, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1, activation = 'sigmoid'))


model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 100)

final = model.predict(space_data_te)
target = 'Transported'
submission[target] = final

submission['Transported'] = np.rint(submission['Transported'])

submission['Transported'] = submission['Transported'].astype('bool')

submission.to_csv('eight_submission.csv', index=False, header=1)