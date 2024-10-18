# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
import numpy as np
import pandas as pd
import math
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import uniform
from datetime import datetime
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

train = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')

# 삭제 전 ID 저장
train_idx = train['id']
test_idx = test['id']

train = train.drop(columns=['id'])
test = test.drop(columns=['id'])

# 데이터 준비
y = train['loan_status']  # 타겟 변수
X = train.drop('loan_status', axis=1)  # 특징 변수


# %%
# 범주형 및 수치형 변수 정의
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
numerical_cols = [col for col in X.columns if col not in categorical_cols]


## train의 데이터 분포 확인 
num_cols = len(numerical_cols)
cols = 3 
# 반올림을 통해 그래프 누락 방지 (10/3 ->4)
rows = math.ceil(num_cols / cols) 

fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))

# reshape(-1)은 단지 프로그래밍 상의 편의를 위해 1차원 배열로 바꾸는 것일 뿐, 플롯의 시각적 배치는 **행렬 구조(2x3)**가 유지
# reshape(-1)을 통해 반복문에서 [i][j] -> [i]로 단순하게 변형할 수 있음 
# 1 2 3 4 5 6 으로 데이터 값을 가지나, 실제 배열은 2 * 3으로 배치 됨 
axes = axes.reshape(-1)

# kdeplot을 통한 loan_status에 따른 컬럼별 데이터 분포 확인
for i, column in enumerate(numerical_cols):
    sns.kdeplot(train[train['loan_status'] == 0][column], label='loan_status = 0', shade=True, ax=axes[i])
    sns.kdeplot(train[train['loan_status'] == 1][column], label='loan_status = 1', shade=True, ax=axes[i])

# %% [markdown]
# ### Person_age, cb_person_cred_hist_length의 경우 크게 차이 없어보임 
# ### Person_income, person_emp_length, loan_amnt의 경우 애매함 
# --> t- test를 통한 평균 비교 후 유의미한 차이 없으면,, 날려도 되지 않을까 ..

# %% [markdown]
# ## 변수 전처리
#   - 변수 생성
#     1. df['amnt_income'] : 연 소득 대비 대출금액 비율 *percent_income : 연 소득 대비 상환 비율
#     2. df['loan_debt'] : 대출에 대한 이자금액
#     3. df['percent_debt_income'] : 연 수입 대비 이자 비율 
#     4. df['loan_debt_log'] : 대출에 대한 이자 금액의 지수변환 -> loan_depth의 왜도가 +이므로 지수 변환 시도
#     
#     
#   - 변수 제외 
#     1. person_age :  t-test 결과 loan_status 에 따른 평균 유의차 없음
#     2. cb_person_cred_hit_length :  t-test 결과 loan_status 에 따른 평균 유의차 없음

# %%

# 새로운 변수 생성
def new(df):
    df['amnt_income'] = df['loan_amnt'] / df['person_income'] # 연 소득 대비 대출 비율, <-> percent_income : 연 소득 대비 상환 비율(원리금 일듯)
    df['loan_debt'] = df['loan_amnt'] * (1 + df['loan_int_rate']) # 대출에 대한 이자금액
    df['percent_debt_income'] = df['loan_debt'] / df['person_income'] # 연 수입 대비 이자 비율 
    df['loan_debt_log'] = np.log1p(df['loan_debt']) # 대출에 대한 이자 금액의 지수변환
    

new(train)
new(test)

# 데이터 준비
y = train['loan_status']  # 타겟 변수
X = train.drop('loan_status', axis=1)  # 특징 변수

# 학습 및 검증 데이터셋 나누기
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 범주형 및 수치형 변수 정의
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# %%
## train의 데이터 분포 확인 

num_cols = len(numerical_cols)
cols = 3 
# 반올림을 통해 그래프 누락 방지 (10/3 ->4)
rows = math.ceil(num_cols / cols) 

fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))

# reshape(-1)은 단지 프로그래밍 상의 편의를 위해 1차원 배열로 바꾸는 것일 뿐, 플롯의 시각적 배치는 **행렬 구조(2x3)**가 유지
# reshape(-1)을 통해 반복문에서 [i][j] -> [i]로 단순하게 변형할 수 있음 
# 1 2 3 4 5 6 으로 데이터 값을 가지나, 실제 배열은 2 * 3으로 배치 됨 
axes = axes.reshape(-1)

# kdeplot을 통한 loan_status에 따른 컬럼별 데이터 분포 확인
for i, column in enumerate(numerical_cols):
    sns.kdeplot(train[train['loan_status'] == 0][column], label='loan_status = 0', shade=True, ax=axes[i])
    sns.kdeplot(train[train['loan_status'] == 1][column], label='loan_status = 1', shade=True, ax=axes[i])

# %% [markdown]
# ### T-test 평균 유의차 검정
# - loan_status에 따라 평균 차이 여부 검정 -> 유의하지 않은 변수 제외

# %%
from scipy import stats

# 두 그룹 평균 차이에 대한 ttest 시행 함수
def ttest_ind(df, feature, value='loan_status'):
    t, p = stats.ttest_ind(df.loc[df[value] == 0, feature],
                       df.loc[df[value] == 1, feature],
                       equal_var=True)
    print(f't-statistic: {t}, p-value: {p}')
    if p >= 0.05:
      print("귀무가설 채택")
    else:
      print("귀무가설 기각 ==> 대립가설 채택")


# 'person_age'와 'cb_person_cred_hist_length'는 유의하지 않아서 제외 
ttest_ind(train, 'person_age'), ttest_ind(train, 'person_emp_length'), ttest_ind(train, 'loan_amnt'), ttest_ind(train, 'person_income'), ttest_ind(train, 'cb_person_cred_hist_length')


# %%
# 특정 문자열을 수치형 컬럼에서 제거
remove_strings = ['person_age', 'cb_person_cred_hist_length', 'loan_debt']
numerical_cols = [item for item in numerical_cols if item not in remove_strings]

# %% [markdown]
# ### 로지스틱 회귀분석
# - 변수별 loan_status에 미치는 영향성 확인

# %%
import statsmodels.api as sm
from statsmodels.formula.api import logit

# 로지스틱 회귀분석
model = logit('loan_status ~ person_income + person_home_ownership+person_emp_length+loan_intent+loan_grade+loan_amnt+\
loan_int_rate+loan_percent_income+amnt_income+percent_debt_income+ loan_debt_log', data = train).fit()
print(model.summary())


# %% [markdown]
# ### 다중 T-test 

# %%
!pip install pingouin -qq 

# %%
import pingouin as pg

# loan_grade가 높을수록 loan_status 1의 비율 차이가 있으나, 낮을수록 그렇게 차이는 없는듯 하다.
pg.pairwise_tukey(data=train, dv='loan_status', between='loan_grade')

# %% [markdown]
# ## 데이터 분포 시각화 
# - 그래프에 대한 해석
#     * 위 그래프 확인 결과, 00000
# 
# - 주로 사용하는 시각화
#     + histplot
#     + countplot
#     + kdeplot
#     + boxplot
#     + heatmap -> corr()

# %%
# 컬럼별 상관관계 확인 -> 다중 공선성 제어를 위함 
tra_corr = train[numerical_cols].corr()

# 상관관계 수준 heatmap으로 확인
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(tra_corr,annot=True, fmt=".2f", cmap = 'Reds', linewidth=.3, )

# %%
# Box plot : loan_status에 따른 수준 비교 
def box_status(df, *features):
    rows = math.ceil(len(features) / 3)
    fig, ax = plt.subplots(nrows=rows, ncols=3, figsize=(15, rows * 5))  # 크기 조정
    ax = ax.reshape(-1)  # 1차원으로 변환
    
    for i, col in enumerate (features):
        sns.boxplot(data = df, x='loan_status', y= col, width=.5, linewidth=.55, ax=ax[i])
        
    plt.tight_layout()
    plt.show()
    
box_status(train,'loan_amnt', 'percent_debt_income', 'amnt_income', 'loan_percent_income')


# %%
import pingouin as pg

# loan_grade가 높을수록 loan_status 1의 비율 차이가 있으나, 낮을수록 그렇게 차이는 없는듯 하다.
res = pg.pairwise_tukey(data=train, dv='loan_amnt', between='loan_status')

print(res)


# %%
print(pg.pairwise_tukey(data=train, dv='loan_amnt', between='loan_status'))

# %%
print(pg.pairwise_tukey(data=train, dv='percent_debt_income', between='loan_status'))

# %%
print(pg.pairwise_tukey(data=train, dv='amnt_income', between='loan_status'))

# %%
print(pg.pairwise_tukey(data=train, dv='loan_percent_income', between='loan_status'))

# %% [markdown]
# ## 대출 거절의 경우 부적적 수치가 높음
# - 원리금 비율이 수입 대비 높은 편이라고 할 수 있다.

# %%
g = sns.FacetGrid(train, col='loan_status', height=4, aspect=1.5)
g.map(sns.countplot, 'person_emp_length')

# %%
# loan_grade 에 따른 loan_status 차이
fig, ax = plt.subplots(figsize=(8,6))
sns.countplot(train, x='loan_grade', hue='loan_status')
plt.show()

# %%
from scipy.stats import probplot
from scipy.stats import shapiro


fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 정규성 확인을 위한 Q-Q Plot
probplot(train['loan_debt_log'], dist="norm", plot = axes[0]);
sns.histplot(train['loan_debt_log'], axes=axes[1]);

# %%
# 정규성 여부 검증 
# n이 충분히 크기 때문에 정규성 여부 검증은 필요 없으나, 로그변환 효과를 확인하기 위해 정규성 검증 시행
from scipy import stats

# 샘플 데이터
data = train['loan_debt_log']  # 예시 데이터

# Shapiro-Wilk 테스트
shapiro_stat, shapiro_p_value = stats.shapiro(data)

print(f'Statistic: {shapiro_stat}, p-value: {shapiro_p_value}')

# p-value가 0.05보다 작으면 정규성이 아닌 것으로 추정
if shapiro_p_value < 0.05:
    print("정규성을 만족하지 않습니다.")
else:
    print("정규성을 만족합니다.")

# %%
# density -> 선그래프, 정규분포 여부 시각적으로 대략 확인 가능
# hist는 pandas에서 제공하는 기능임, kde와 유사하므로 패스
# ax = train["person_age"].hist(bins=17, density=True, stacked=True, color='teal', alpha=0.6)
# train["person_age"].plot(kind='density', color='teal')
# ax.set(xlabel='Age')
# plt.xlim(-10,85)
# plt.show()

# %% [markdown]
# ## ML 성능 평가

# %%
# # (여기서 다른 전처리 작업 추가 가능)

# 전처리 및 모델 파이프라인 설정
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # 수치형 컬럼에 대해 StandardScaler 적용
        # ('cat', OrdinalEncoder(), categorical_cols)  # 범주형 컬럼에 대해 OrdinalEncoder 적용 0.88
         ('cat', OneHotEncoder(drop='first'), categorical_cols)  # 범주형 컬럼에 대해 OneHotEncoder 적용
        
    ]
)

logistic_pipeline = make_pipeline(
    preprocessor,
    LogisticRegression(solver='liblinear')  # 로지스틱 회귀 모델
)

# 하이퍼파라미터 튜닝 (ROC-AUC 스코어 기준)
param_distributions = {
    'logisticregression__C': uniform(0.01, 10),  # 정규화 강도
    'logisticregression__penalty': ['l1', 'l2']  # 정규화 방식
}

random_search = RandomizedSearchCV(
    logistic_pipeline, 
    param_distributions, 
    n_iter=100, 
    cv=2, 
    random_state=42, 
    n_jobs=-1, 
    scoring='roc_auc'  # ROC-AUC를 기준으로 스코어링
)

# 모델 학습
random_search.fit(X_train, y_train)

# 검증 세트에 대한 예측 확률 계산
y_val_probs = random_search.predict_proba(X_val)[:, 1]  # 클래스 1(양성 클래스)에 대한 확률 추출

# ROC-AUC 점수 계산 및 출력
roc_auc = roc_auc_score(y_val, y_val_probs)
print(f'ROC-AUC 점수: {roc_auc:.4f}')

# ROC 곡선 계산
fpr, tpr, thresholds = roc_curve(y_val, y_val_probs)

# ROC 곡선 그리기
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='red', linestyle='--')  # 랜덤 추측선
ax.set_xlabel('False Positive Rate (FPR)')  # 위양성 비율
ax.set_ylabel('True Positive Rate (TPR)')  # 진양성 비율
ax.set_title('ROC Curve')
ax.legend(loc='lower right')

# 그리드 제거
ax.grid(False)

# 검증 데이터에 대한 예측 값 생성
y_val_preds = random_search.best_estimator_.predict(X_val)

# 혼동 행렬 계산
cm = confusion_matrix(y_val, y_val_preds)

# ROC 곡선 안에 혼동 행렬 삽입
ax_inset = inset_axes(ax, width="30%", height="30%", loc="center right")  # 위치 조정 가능
ConfusionMatrixDisplay(cm).plot(ax=ax_inset, colorbar=False)
ax_inset.grid(False)  # 혼동 행렬에 그리드 제거
ax_inset.set_title('Confusion Matrix')  # 작은 창에 제목 추가

# 그래프 보여주기
plt.show()

# 테스트 데이터 예측 및 제출 파일 저장
test = test.loc[:, categorical_cols + numerical_cols]
now = datetime.now()

# 테스트 데이터에 대한 예측 수행
submission['loan_status'] = np.round(random_search.best_estimator_.predict_proba(test)[:, 1], 2)
submission.to_csv(f'result_{now.strftime("%Y-%m-%d_%H-%M-%S")}.csv', index=False)


# %% [markdown]
# ## 통계분석
# - t-test, AVOVA, 회귀분석, 로지스틱 회귀분석
# - 결과 해석, 통계량

# %% [markdown]
# ## 분석의 주요 목적
# - 머신러닝 알고리즘을 수행할 때, 가장 중요한 컬럼을 선정
# - 기본 베이스 모델 : 전체 컬럼 사용
# - 해야할 일
#     + 파생변수 생선 -> 컬럼 증가
#     + 수치 데이터 -> 범주형 변환
#     + train, test 각각 데이터 적용
# 


