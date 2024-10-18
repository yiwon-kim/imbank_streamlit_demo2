# Loan Approval Prediction
##분석 목적
1. 변수 분석을 통한 대출 승인 여부 예측
2. ML Score 올리기
3. Python 활용 능력 쌓기

## 변수 설명
### Basic Variables
person_age: Applicant’s age in years.
person_income: Annual income of the applicant in USD.
person_home_ownership: Status of homeownership (e.g., Rent, Own, Mortgage).
person_emp_length: Length of employment in years.
loan_intent: Purpose of the loan (e.g., Education, Medical, Personal).
loan_grade: Risk grade assigned to the loan, assessing the applicant’s creditworthiness.
loan_amnt: Total loan amount requested by the applicant.
loan_int_rate: Interest rate associated with the loan.
loan_status: The approval status of the loan (approved or not approved).
loan_percent_income: Percentage of the applicant’s income allocated towards loan repayment.
cb_person_default_on_file: Indicates if the applicant has a history of default ('Y' for yes, 'N' for no).
cb_person_cred_hist_length: Length of the applicant’s credit history in years.

### Additional Variables
amnt_income: 연 소득 대비 대출금액 비율
percent_income: 연 소득 대비 상환 비율
loan_debt: 대출에 대한 이자금액
percent_debt_income: 연 수입 대비 이자 비율
loan_debt_log: 대출에 대한 이자 금액의 지수변환 -> loan_depth의 왜도가 +이므로 지수 변환 시도

### Excluded Variables
person_age: t-test 결과 loan_status 에 따른 평균 유의차 없음
cb_person_cred_hist_length: t-test 결과 loan_status 에 따른 평균 유의차 없음
