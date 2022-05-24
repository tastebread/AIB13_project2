import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from matplotlib import rc
import plotly.express as px
import scipy.stats as stats
import mplcyberpunk
from pandas_profiling import ProfileReport
from scipy.stats.contingency import association
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from category_encoders import OneHotEncoder
from category_encoders import OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import warnings
plt.style.use('cyberpunk')#시각화 테마

warnings.filterwarnings(action='ignore') #경고문구 삭제
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')
"""
1.PassengerId -> 각 승객에 대한 고유 ID
각 ID는 승객이 함께 하는 여행그룹 gggg_pp 에서 gggg를 나타내며, pp는 그룹내의 번호(그룹내 사람들은 가족구성원이지만 항상 그런것은 아니다)
2.HomePlanet -> 승객이 출발한 행성(출발지), 일반적으로 영구 거주 행성입니다
3.CryoSleep -> 승객이 항해 기간 동안 동면 여부를 나타냄 ( 승객은 무조건 객실에만 있는다)
4.Cabin -> 승객이 머물고 있는 객실번호 를 나타낸다
deck/num/side 여기서 side는 P(Port),S(starboard) 이다
5.Destination - 승객이 출발할 행성(도착지 행성)
6.Age -> 승객의 나이, VIP -> 승객이 항해 중 특별 VIP 서비스를 지불했는지 여부
7.RoomService,FoodCourt,ShoppingMall - 승객이 이용한 고급 편의 시설에 이용여부
8.Spa,VRDeck -> 고급편의시설에 대한 청구내용과(Spa), 금액(VRDeck)
9.Name -> 승객의 성과 이름
10.Transported - > 승객이 다른 차원으로 이송되었는지 여부 (예측하려는 대상 target)
""" #train 데이터셋 컬럼 설명
df = [train,test] #train,test 데이터를 리스트로 만들어준다
#같은 타입의 컬럼들끼리 묶어주기
df_num = train[['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Transported']]
df_cat = train[['PassengerId','HomePlanet','CryoSleep','Cabin','Destination','VIP','Name']]
#print(train.dtypes )
"""-----------------------------------------------------------------------------------------"""
#숫자형 컬럼의 분포도 알아보기
"""
sns.pairplot(df_num, hue='Transported')
#pairplot -> 각 컬럼별 데이터에 관해 상관관계나 분류적 특성을 보여줌
# hue -> 카테고리 값에 따라 색상 변경 가능
# vars -> 내가 원하는 몇개의 컬럼만 볼수있음
plt.savefig('numerical var.png')
"""
#숫자형 컬럼의 상관계수 알아내기
"""
plt.subplots(figsize=(20,15))
plt.title('numerical columns corr')
sns.heatmap(df_num.corr(), cmap='cool',annot=True, annot_kws={'fontsize': 15},square =True,linewidths=5)
plt.savefig('numerical heatmap.png')
#heatmap 파라미터
#annot -> 각 셀의 값 표시 여부
#linewidth -> 굵기 설정
#annot_kws -> 글씨 사이즈
시각화를 통해 알아낸것
1.FoodCourt 는 Spa,VRDeck 와 밀접한 관계가 있다
2.의외로 RoomService,Spa,VRDeck 은 타겟과 영향이 없을 수 있음(지표로만 봤을때)
"""

#서비스 시설을 3종류로 나눠보기
for datasets in df:
    datasets['Premium'] = datasets['RoomService'] + datasets['Spa'] + datasets['VRDeck']
    datasets['Basic'] = datasets['FoodCourt'] + datasets['ShoppingMall']
    datasets['All_Services'] = datasets['RoomService'] + datasets['Spa'] + datasets['VRDeck'] + datasets['FoodCourt'] + datasets['ShoppingMall']

df_num = train[['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Premium','Basic','All_Services','Transported']]

#3개의 새로운컬럼 시각화 (Premium,Basic,All_Services)
"""
sns.pairplot(df_num,hue='Transported',vars=['Premium','Basic','All_Services'])
plt.title('Premium,Basic,All_Services')
plt.savefig('ServicesSet.png')
"""
"""-----------------------------------------------------------------------------------------"""
#EDA, 특성공학
#1.Age 컬럼을 groupby 를 이용해서 평균값을 계산 AgeGroup 라는 피쳐로 만들어보기

for datasets in df:
    datasets['Age'] = datasets['Age'].fillna(datasets['Age'].mean())
    datasets['AgeGroup'] = pd.cut(datasets['Age'],6) #pd.cut -> 특정 구간을 나누어서 다양한 그룹으로 만들 수있다
    # 파라미터 -> (데이터,구간,labels) % labels 는 따로 지정해주지 않으면 구간의 나눈 기준이 레이블명이됨
#print(train[['AgeGroup','Transported']].groupby(['AgeGroup'], as_index=False).mean())
#.groupby -> 통계 또는 집계 결과를 얻기 위해 사용하는것 (AgeGroup의 평균을 이용해서 나타낸다)
#as_index - > groupby를 사용하면 기본으로 그룹 라벨이 index 가 되는데 사용하고싶지 않은경우 False

#AgeGroup컬럼의 결측치값을 평균으로 넣어준다
"""
13세이하 -> 0, 13세 이상 26세 이하 -> 1, 26세 이상 39세 이하 -> 2, 39세이상 52세이하 -> 3
52세 이상 65세 이하 -> 4, 65세 이상 79세 이하 -> 5, 79세 이상 -> 6

결과 : 나눠주기 전과 후 결과 값이 조금 증가했다
"""
for datasets in df:
    datasets.loc[datasets['Age'] < 13, 'AgeGroupNum'] = 0
    datasets.loc[(datasets['Age'] >=13) & (datasets['Age'] <=26), 'AgeGroupNum'] = 1
    datasets.loc[(datasets['Age'] >= 26) & (datasets['Age'] <= 39), 'AgeGroupNum'] = 2
    datasets.loc[(datasets['Age'] >= 39) & (datasets['Age'] <= 52), 'AgeGroupNum'] = 3
    datasets.loc[(datasets['Age'] >= 52) & (datasets['Age'] <= 65), 'AgeGroupNum'] = 4
    datasets.loc[(datasets['Age'] >= 65) & (datasets['Age'] <= 79), 'AgeGroupNum'] = 5
    datasets.loc[datasets['Age'] > 79, 'AgeGroupNum'] = 6
print(train[['AgeGroupNum','Transported']].groupby(['AgeGroupNum'],as_index=False).mean())

#나머지 숫자형 변수 결측치 채우기
for datasets in df:
    datasets['RoomService'] = datasets['RoomService'].fillna(datasets['RoomService'].median())
    datasets['FoodCourt'] = datasets['FoodCourt'].fillna(datasets['FoodCourt'].median())
    datasets['ShoppingMall'] = datasets['ShoppingMall'].fillna(datasets['ShoppingMall'].median())
    datasets['Spa'] = datasets['Spa'].fillna(datasets['Spa'].median())
    datasets['VRDeck'] = datasets['VRDeck'].fillna(datasets['VRDeck'].median())

"""
plt.subplots(figsize=(20,15))
plt.title('숫자형 컬럼 상관계수(결측치채움)')
sns.heatmap(df_num.corr(), cmap='cool',annot=True, annot_kws={'fontsize': 15},square =True,linewidths=5)
plt.savefig('numerical heatmap(fillna on).png')
"""#숫자형 변수들 상관관계 알아보기(결측치 다채운값)
"""
#EDA, 특성공학 확인전 특성중요도 확인해보기
target='Transported'
features=train.drop(columns=[target]).columns

train,val = train_test_split(train, train_size=0.8,test_size=0.2,stratify=train[target],random_state=42)
X_train = train[features]
y_train = train[target]

X_val = val[features]
y_val = val[target]

pipe = make_pipeline(
    OrdinalEncoder(),
    SimpleImputer(),
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
)
print(pipe)

#train 셋 검증 정확도
pipe.fit(X_train,y_train)
print(f'검증 정확도 : {pipe.score(X_val,y_val)}')

#특성중요도
rf = pipe.named_steps['randomforestclassifier']
importances = pd.Series(rf.feature_importances_, X_train.columns)

n=20
plt.figure(figsize=(30,20))
plt.title(f'특성중요도 탑{n}')
importances.sort_values()[-n:].plot.barh()
plt.savefig('importances-20.png')
""" #머신러닝 적용및 특성중요도 계산

"""
def QQplot(df,col):
    fig, axes=plt.subplots(1,2, figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(x=df[col],kde=True)

    plt.subplot(1,2,2)
    stats.probplot(x=df[col].dropna(), dist='norm',plot=plt)
    plt.tight_layout()
    plt.show()

df_services = train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Premium', 'All_Services', 'Transported']]
#QQplot(df_services,'RoomService')
"""
#print("베이스라인 모델 : {}".format(train['Transported'].mean()))

#범주형 변수 결측치 채우는 가장 일반적인 방법 라벨의 모드를 이용
df_cat = train[['HomePlanet','CryoSleep','Cabin','Destination','VIP','AgeGroupNum']]

#범주형 변수 시각화 함수
def Catplot(df,x,y):
    plt.subplots(1,2, figsize=(14,5))
    plt.subplot(1,2,1)
    sns.countplot(x=df[x].dropna(), hue=df[y])

    plt.subplot(1,2,2)
    plt.ylim(0,1)
    sns.lineplot(x=df[x],y=df[y], data=df, ci=None,linewidth=3, marker='o')
train.drop(columns=['Name'])

plt.figure(figsize=(10,5))
sns.histplot(data=train,x='HomePlanet',hue='Transported',bins=7,binwidth=1,kde=True)
#파라미터 data -> 데이터셋, x-> x축에 들어갈 컬럼 hue-> 특성에 따라 색상을 다르게 할수있음
#bins -> 구간의 갯수 정하기 , binwidth -> 구간의 크기를 수동으로 정하기
#kde -> 커널밀도함수 표시여부 (True가 디폴트값)
plt.savefig('HomePlanet-Transported')
"""
1.대부분 지구에서 출발한다라는것을 알수있다
결측치부분을 지구로 채워주면 좀더 높아질 가능성이 있다
 """ #homeplanet 과 이송률에 관한 관계
train['HomePlanet'].fillna('Europa',inplace=True)

"""
Catplot(train,'HomePlanet','Transported')
plt.savefig('Homeplanet-Transported[end].png')
""" #Homeplanet 시각화 완료
print(train.dtypes)

train['Cabin'].fillna(method='ffill',inplace=True)
print(train['Cabin'].isnull().sum())

"""-----------------------------------------------------"""
target='Transported'
features = train.drop(columns=[target]).columns

train, val = train_test_split(train,stratify=train[target],train_size=0.8, test_size=0.2,random_state=42)


X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]

X_test = test[features]
"""
pipe = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    SelectKBest(f_regression, k=20),
    Ridge(alpha=1.0)
)

k = 3
scores = cross_val_score(pipe, X_train, y_train, cv=k,
                         scoring='neg_mean_absolute_error')

print(f'MAE ({k} folds:', -scores)
print(f'평균 : {-scores.mean()} 표준편차 : {-scores.std()}')


pipe = make_pipeline(
    OrdinalEncoder(),
    SimpleImputer(),
    DecisionTreeRegressor()
)

depth = range(1,30,2)

ts, vs = validation_curve(
    pipe, X_train, y_train
    , param_name='decisiontreeregressor__max_depth'
    , param_range=depth, scoring='neg_mean_absolute_error'
    , cv=3
    , n_jobs=-1
)

train_scores_mean = np.mean(-ts, axis=1)
validation_scores_mean = np.mean(-vs, axis=1)
""" #k fold 검정
"""
fig,ax = plt.subplots()

#훈련세트 검증곡선
ax.plot(depth, train_scores_mean,label='train error')
#검증세트 검증곡선
ax.plot(depth, validation_scores_mean,label='validation error')

#이상적인 max_depth
ax.vlines(5,0, train_scores_mean.max(), color='blue')

#그래프 셋팅
ax.set(title='validation curve',
       xlabel = 'Model complexity(max_depth)', ylabel='MAE')
ax.legend()
fig.dpi = 100
plt.savefig('validation curve.png')

#결과 : max_depth = 5  부근에서 설정해주면 과적합을 막고 일반화 성능을 지킬 수 있을것같다
""" #파이썬 그래프 셋팅



#1.하이퍼파라미터 튜닝
"""
사이킷런을 사용해서 하이퍼파라미터를 최적화 할수 있다
머신러닝 모델을 만들때 중요한 것은 최적화,일반화 이다
최적화 : 훈련 데이터로 더 좋은 성능을 얻기 위해 모델을 조정하는 과정
일반화 : 학습된 모델이 처음 본 데이터에서 얼마나 좋은 성능을 내는지 이야기 하는것
모델의 복잡도를 높이는 과정에서 훈련/검증 세트의 손실이 함께 감소하는 시점은 과소적합(underfitting) 되었다고 합니다.
훈련데이터의 손실은 계속 감소하는데 검증데이터의 손실은 증가하는 때가 있습니다. 이때 우리는 과적합(overfitting) 되었다고 합니다
이상적인 모델은 과소적합과 과적합 사이에 존재한다
"""
#1.2 Randomized Search CV
"""
여러 하이퍼파라미터의 최적값을 찾기 위한 randomizedSearchCV 사용
하이퍼파라미터 : 모델 훈련중에 학습이 되지 않는 파라미터로 사용자가 직접 지정해 주어야한다
현실적으로 일일이 수작업으로 정해주는것은 어렵고 최적의 하이퍼파라미터 조합을 찾아주는 도구를 사용해야한다 (사이킷런에는 두가지 툴이있음)

GridSearchCV : 검증하고 싶은 하이퍼파라미터 수치를 정해주고 그 조합을 모두 검증함
RandomizedSearchCV : 검증하려는 하이퍼파라미터들의 범위를 지정해주면 무작위로 값을 지정해 그 조합을 모두 검증
"""
"""
pipe1 = make_pipeline(
    OneHotEncoder(use_cat_names=True)
    , SimpleImputer()
    , StandardScaler()
    , SelectKBest(f_regression)
    , Ridge()
)
disit = {
    'simpleimputer__strategy' : ['mean','median'],
    'selectkbest__k' : range(1, len(X_train.columns)+1),
    'ridge__alpha' : [0.1,1,10],
}

clf = RandomizedSearchCV(
    pipe1,
    param_distributions=disit,
    n_iter=50,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=1,
    n_jobs=-1
)

clf.fit(X_train,y_train)

print(f'최적 하이퍼파라미터 : {clf.best_params_}')
print('MAE: ', -clf.best_score_)
"""

#기준모델로 최다빈도를 사용할겨우

major = y_train.mode()
y_pred = [major] * len(y_train)
print(f'trainning accuracy : {accuracy_score(y_train,y_pred)}')



#검증 파이프라인 만들어주기

pipe_val = make_pipeline(
    OneHotEncoder(),
    SimpleImputer(),
    StandardScaler(),
    RandomForestClassifier(n_estimators=50,n_jobs=-1,random_state=42)
)

model = pipe_val.fit(X_train,y_train)
print(f'훈련 정확도 : {pipe_val.score(X_train,y_train)}')
print(f'검증 정확도 : {pipe_val.score(X_val,y_val)}')


#케글 제출 코드

y_pred = model.predict(X_test)
target = 'Transported'
submission[target] = y_pred

submission.to_csv('first_submission.csv', index=False, header=1)
