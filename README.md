# *Classifying Credit Card Users: A Machine Learning Analysis*
1. 프로젝트 소개
2. EDA 및 전처리
3. ML Modeling
## 프로젝트 소개
### 배경
- 신용카드 사용률은 매년 증가하고 있으며 개인 이용금액 또한 증가하고 있음.
- 금리 상승, 신용카드 연체율 급증으로 인해 신용카드사의 자산건전성 약화와 위험부담이 증가하고 있음.
- 국내 빅테크 기업 주도 아래 후불 결제 시장이 성장하면서 연체율 관리가 매우 중요해짐.
- 신용카드사는 신용등급으로 연체 가능성을 판단하기에 신용등급 산정은 매우 중요함.
### 목적
- 카드 대금 연체 집단의 정보를 통해 연체 정도를 예측할 수 있는 알고리즘을 개발하고 건전한 금융시장 유지에 도움이 되는 인사이트를 제공한다.
## EDA 및 전처리
#### 1. Data 분포 확인
- 각 Feature별 데이터 분포를 확인 결과 Laber값에 유의미한 차이를 만드는 Feature 존재하지 않음
- Creidt(0, 1, 2)별 데이터 분포 확인 결과 각 Credit에 따라 비슷한 분포를 보임
- Income_total 컬럼의 경우 boxPlot 시각화 확인 결과 이상치 데이터 다수 존재
#### 1. Feature 제거
- Child_num과 family_size 관계
>> 가족 수 6 -> 자녀 수 4.5 <br>
>> 가족 수 7 -> 자녀 수 5 <br>
>> 가족 수 9 -> 7 <br>
>> 가족 수 15 -> 14 <br>
>> 가족 수 20 -> 19
- 두 Feature의 경우 상관계수 $0.89$로 높은 상관도를 보이며, 가족 수 증가에 따른 자녀 수가 증가하는 선형관계로서 다중공선성을 막기 위해 Family_size 컬럼 삭제
- FLAG_MOBIL Feature의 경우 고유값이 하나 존재하기 때문에 삭제
#### 2. 이상치 제거
- Child_num Feature Data의 경우 자녀 수 5명 이상 데이터가 전체 데이터 중 $0.1%$이기 때문에 이상치로 판단 후 제거
#### 3. 연속형 자료의 변환
- DYAS_EMPLOYED Feature의 경우 양수값의 경우 수치와 상관없이 무직자이기 때문에 0으로 변환
- 연속형 자료 전체 데이터를 음수 기준에서 양수 기준으로 변환
#### 4. 결측값 확인
- Occupy_type Feature Data에서 결측값을 확인
#### 5. Feature 범주화 및 단위 조정
- 범주형 Data는 LabelEncoder로 변환
- 연속형 Data는 단위별 단순 변환
## ML Modeling
- Log_loss : $-\frac{1}{N}\sum_{i=1}^{N} yi \times \log(p(yi))+(1+yi) \times \log(1-p(yi))$
- Target의 실제 값에 대한 Predict probability를 log변환한 값의 평균
- 자연로그의 특성상 probability가 0에 가까워지는 경우 그 값이 음의 무한대로 수렴하기 때문에 예측실패한  데이터의 probability에 따라 가중치가 더해짐.
- 불균형 데이터의 경우 다수를 차지하는 데이터에 대한 예측은 좋고 반대의 경우 예측이 낮아지는 경우가 많은데, 
예측의 실패정도에 가중치가 주어지기 때문에 평가지표로 적당하다 판단되어 선정
#### 모델 성능 결과 확인
- [XGBoost LightGBM CatBoost RandomForest DecisionTree] 5개 모델 성능 확인 결과
<img width="412" alt="1" src="https://user-images.githubusercontent.com/125957806/233585625-329718e5-bb87-455d-81bd-5b3cab24de44.png">

- 성능 개선을 위한 파생변수 생성

>> income_occupy:  소득에 따른 직업 유형 <br>
>>car_reality:  자산 소유 여부<br>
>>income_wage:  연소득<br>
>>employed_wage:  근로소득<br>
>>card_begin_before_employed:  카드 발급일 기준 근로 여부<br>
>>before_EMPLOYED:  미취업기간<br>
>>income_total_beforeEMP_ratio:  취업 전 소득<br>
>>DAYS_BIRTH_m:   태어난 월<br>
>>DAYS_BIRTH_w:   태어난 주<br>
>>DAYS_EMPLOYED_m:   고용된 월<br>
>>DAYS_EMPLOYED_w:   고용된 주<br>
>>ability:  연령 /   근무일 대비 소득<br>
>>income_mean:      가족 수를 고려한 소득<br>
- 파생변수 생성 후 모델 성능 확인
<img width="421" alt="2" src="https://user-images.githubusercontent.com/125957806/233586671-6ded162f-f038-4d6f-9048-dff0a1ae0678.png">

#### 문제점 발견
- 파생 변수를 추가했을 때 오히려 성능이 떨어짐
- 다양한 방식으로 columnm 선택의 변화를 주었으나, 성능 개선이 이루어지지 않음
#### 중복 데이터 확인
- Begin_month 는 같지만 , Credit 이 다른 경우
- Credit 은 같지만 , Begin_month 가 다른 경우
- <img width="368" alt="3" src="https://user-images.githubusercontent.com/125957806/233587070-acbffeaa-8921-4885-99f2-47aba4c495d9.png">

#### 고유 ID 컬럼 생성
- 한 사람이 여러 카드를 발급받았다는 가정 하에, begin_month와 credit를 제외한 모든 컬럼을 합쳐 한 사람을 식별하는 고유 ID 컬럼 생성
- 고유 ID 컬럼을 생성 후 모델 성능확인
<img width="275" alt="4" src="https://user-images.githubusercontent.com/125957806/233587396-6761a59a-7921-4462-bfcc-881a30ae2191.png">

#### 중복 데이터 삭제
- Credit은 같지만, Begin_month가 다른 경우 중복 데이터 삭제 후 성능 확인
<img width="282" alt="5" src="https://user-images.githubusercontent.com/125957806/233788683-0ac2ef9b-83ea-4b65-8eac-18562d5c21ac.png">

#### 중복 데이터 삭제 및 PCA와 Clustering
- 중복 데이터를 삭제 후 PCA를 통해 차원을 축소하고 군집화 형성 후 성능 확인
<img width="280" alt="6" src="https://user-images.githubusercontent.com/125957806/233788772-16218424-b2a2-4fb4-b594-2cbcdfd416c3.png">

#### SMOTE_Oversampling 적용
- 데이터가 적은 Credit 0과 1에 대한 Recall값을 높이기 위해 OverSampling 적용 후 성능 확인
<img width="294" alt="7" src="https://user-images.githubusercontent.com/125957806/233788825-f2a40ce9-ea98-4a3e-98a9-23d7ff4b1dd6.png">

#### 최종 모델
- 최종적으로 선정한 모델과 해당 기법 및 EDA 방법
<img width="355" alt="8" src="https://user-images.githubusercontent.com/125957806/233788881-d13b7ca0-5444-47e2-a429-19d18b49bcfc.png">
