import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import math
import pingouin as pg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats

# 데이터 로드 및 전처리 함수
@st.cache_data
def load_and_preprocess_data():
    # 데이터 로드 및 전처리
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    submission = pd.read_csv('dataset/sample_submission.csv')

    # 불필요한 열 제거
    train = train.drop(columns=['id'])
    test = test.drop(columns=['id'])

    # 새로운 변수 생성
    def create_new_features(df):
        df['amnt_income'] = df['loan_amnt'] / df['person_income']
        df['loan_debt'] = df['loan_amnt'] * (1 + df['loan_int_rate'])
        df['percent_debt_income'] = df['loan_debt'] / df['person_income']
        df['loan_debt_log'] = np.log1p(df['loan_debt'])

    create_new_features(train)
    create_new_features(test)

    return train, test, submission

def box_status(df, *features):
    rows = math.ceil(len(features) / 3)
    fig, ax = plt.subplots(nrows=rows, ncols=3, figsize=(15, rows * 5))  # 크기 조정
    ax = ax.reshape(-1)  # 1차원으로 변환
    
    for i, col in enumerate(features):
        sns.boxplot(data=df, x='loan_status', y=col, width=.5, linewidth=.55, ax=ax[i])
        
    plt.tight_layout()
    st.pyplot(fig)  # Display plot in Streamlit

# KDE plot function
def kdeplot_loan_status(df, feature):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(data=df, x=feature, hue="loan_status", fill=True, common_norm=False, palette="crest", alpha=0.5, linewidth=1)
    st.pyplot(fig)  # Display plot in Streamlit

# Tukey's HSD Test results
def perform_statistical_analysis(df):
    results = {}
    results['loan_amnt'] = pg.pairwise_tukey(data=df, dv='loan_amnt', between='loan_status')
    results['percent_debt_income'] = pg.pairwise_tukey(data=df, dv='percent_debt_income', between='loan_status')
    results['amnt_income'] = pg.pairwise_tukey(data=df, dv='amnt_income', between='loan_status')
    results['loan_percent_income'] = pg.pairwise_tukey(data=df, dv='loan_percent_income', between='loan_status')
    return results

def display_tukey_results(results):
    for feature, res in results.items():
        st.subheader(f"Tukey's HSD Test Results for {feature}")
        st.write(res)

def stream_data():
    # Basic Variable Descriptions
    st.subheader("Basic Variables")
    st.write("**person_age**: Applicant’s age in years.")
    st.write("**person_income**: Annual income of the applicant in USD.")
    st.write("**person_home_ownership**: Status of homeownership (e.g., Rent, Own, Mortgage).")
    st.write("**person_emp_length**: Length of employment in years.")
    st.write("**loan_intent**: Purpose of the loan (e.g., Education, Medical, Personal).")
    st.write("**loan_grade**: Risk grade assigned to the loan, assessing the applicant’s creditworthiness.")
    st.write("**loan_amnt**: Total loan amount requested by the applicant.")
    st.write("**loan_int_rate**: Interest rate associated with the loan.")
    st.write("**loan_status**: The approval status of the loan (approved or not approved).")
    st.write("**loan_percent_income**: Percentage of the applicant’s income allocated towards loan repayment.")
    st.write("**cb_person_default_on_file**: Indicates if the applicant has a history of default ('Y' for yes, 'N' for no).")
    st.write("**cb_person_cred_hist_length**: Length of the applicant’s credit history in years.")

    # Additional Variable Descriptions
    st.subheader("Additional Variables")
    st.write("**amnt_income**: 연 소득 대비 대출금액 비율")
    st.write("**percent_income**: 연 소득 대비 상환 비율")
    st.write("**loan_debt**: 대출에 대한 이자금액")
    st.write("**percent_debt_income**: 연 수입 대비 이자 비율")
    st.write("**loan_debt_log**: 대출에 대한 이자 금액의 지수변환 -> loan_depth의 왜도가 +이므로 지수 변환 시도")

    # Excluded Variable Descriptions
    st.subheader("Excluded Variables")
    st.write("**person_age**: t-test 결과 loan_status 에 따른 평균 유의차 없음")
    st.write("**cb_person_cred_hist_length**: t-test 결과 loan_status 에 따른 평균 유의차 없음")

def main():
    # 데이터 로드
    train, test, submission = load_and_preprocess_data()
    numerical_columns = train.select_dtypes(include=[np.number]).columns.tolist()

    # 사이드바 메뉴
    option = st.sidebar.selectbox("Select an option", 
                                   ["변수 설명", "데이터 로드 및 전처리", "Visualizing", "통계 분석", "모델 학습 및 평가"])

    # Title and Analysis Purpose Display
    if option == "변수 설명":
        st.title("Loan Approval Prediction")
        st.subheader("분석 목적")
        st.write("1. 변수 분석을 통한 대출 승인 여부 예측\n 2. ML Score 올리기 \n 3. Python 활용 능력 쌓기")
        stream_data()
    else:
        # Display only the content for the selected option
        if option == "데이터 로드 및 전처리":
            st.write("### 데이터 로드 및 전처리 완료")
            st.write(train.head())

        elif option == "Visualizing":
            st.subheader("KDE Plot of Loan Amount")
            kdeplot_loan_status(train, 'loan_amnt')  # KDE plot

            st.subheader("Box Plots of Features")
            box_status(train, 'loan_amnt', 'percent_debt_income', 'amnt_income', 'loan_percent_income')  # Box plots

        elif option == "통계 분석":
            # T-test 결과 출력
            st.subheader("loan_status별 loan_amnt 차이")
            t_stat, p_value = stats.ttest_ind(train[train['loan_status'] == 0]['loan_amnt'], 
                                                train[train['loan_status'] == 1]['loan_amnt'])
            st.write(f"T-test 결과: t-statistic={t_stat}, p-value={p_value}")
            
            if p_value < 0.05:
                st.write("두 그룹 간 차이가 유의미합니다.")
            else:
                st.write("두 그룹 간 차이가 유의미하지 않습니다.")

            st.divider()    
            # Perform Tukey's HSD and display results
            st.subheader("loan_status별 변수 수준 차이")
            results = perform_statistical_analysis(train)
            display_tukey_results(results)

        elif option == "모델 학습 및 평가":
            X = train.drop(columns=['loan_status'])
            y = train['loan_status']
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
            numerical_cols = [col for col in X.columns if col not in categorical_cols]

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(drop='first'), categorical_cols)
                ]
            )

            logistic_pipeline = make_pipeline(preprocessor, LogisticRegression(solver='liblinear'))
            logistic_pipeline.fit(X_train, y_train)

            y_val_probs = logistic_pipeline.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_val_probs)

            fpr, tpr, thresholds = roc_curve(y_val, y_val_probs)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
            ax.plot([0, 1], [0, 1], linestyle='--', color='red')
            st.pyplot(fig)

            st.write(f"ROC-AUC 점수: {roc_auc:.4f}")

if __name__ == "__main__":
    main()
