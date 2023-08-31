# 회전기계 진동 신호 기반 이상탐지
이 프로젝트는 회전기계의 진동 데이터에 기반하여 이상 탐지 모델링, 평가 기술을 제공합니다. 회전기계는 산업 분야에서 중요한 장비로 사용되며, 이들의 안정적인 운전을 위해 진동 데이터 모니터링은 필수적입니다. 프로젝트의 코드를 통해 다양한 센서에서 수집된 다변량 시계열 데이터를 활용하여 회전기계의 상태를 실시간으로 평가하고, 이상 징후를 탐지하는데 도움을 줍니다.

## 주요 기능
코드의 데이터 전처리 과정은 데이터를 효율적으로 처리하고 모델 학습에 적합하도록 준비합니다. 선형 보간법을 사용하여 데이터의 Sampling rate를 맞춰주고, 센서 단위로 구성된 데이터를 클래스 별로 분류합니다. 이렇게 구성된 데이터는 모델 학습을 위해 변환됩니다. 이 때, 이상 데이터는 현실에서 드물게 발생하기 때문에 모델을 훈련할 때는 정상 데이터만 사용합니다.

인공지능 모델로 Anomaly transformer라는 이상 탐지 모델을 채택합니다. 이 모델은 각 시간 지점 간의 상관관계를 분석하여 이상이 발생하는 시간 지점을 검출합니다. 이를 위해 전처리된 훈련 데이터를 사용하여 모델을 학습하고, 검증 데이터를 활용하여 훈련 데이터에 과적합되었는지 확인합니다. 또한, Early stopping을 통해 적절한 학습 반복 횟수를 결정합니다.

테스트 단계에서는 훈련된 모델에 테스트 데이터를 입력하여 각 윈도우 별로 이상 점수를 계산합니다. F1-score를 통해 모델의 성능을 평가하고 Confusion matrix를 통해서 시각화합니다. 이를 통해 모델의 성능을 직관적으로 알 수 있습니다. 프로젝트의 코드는 특정 데이터셋에 국한되지 않고, data format을 통일하면 다른 다변량 시계열 데이터셋에도 적용할 수 있도록 설계되었습니다.


### 전처리(Preprocessing)
Preprocessor.py 파일을 실행하여 훈련, 검증, 시험 데이터를 전처리합니다. Preprocessing 함수는 데이터 위치, 데이터 클래스 개수, 센서 개수, 측정 길이, 샘플링 주기를 입력으로 받아서, 순서대로 훈련, 검증, 시험 데이터를 출력합니다. 프로그램은 .csv 파일을 불러와 sensor_list 변수에 저장한 후, 측정 길이와 샘플링 주기에 맞게 배열을 생성하여 보간법을 실행합니다. 보간법은 각 센서별로 적용되고, 그 결과는 y_new 리스트에 저장됩니다. 데이터는 클래스별로 재구성되어 concated_dfs에 저장되며, 변환된 데이터는 torch.DataLoader를 사용하여 모델 학습에 적합한 형태로 변환됩니다.

### 이상 탐지 모델(Anomaly Transformer)
AnomalyTransformer.py 파일의 AnomalyTransformer 클래스를 사용하여 다변량 시계열 이상 탐지 모델을 구현합니다. 이 모델은 각 시간 지점의 상관관계를 분석하여 정상 데이터의 패턴을 학습합니다. 모델은 윈도우 크기, 센서 개수, 임베딩 차원, 멀티 헤드 개수, 레이어 개수를 입력으로 받습니다. AnomalyTransformer를 실행하면 DataEmbedding 클래스를 사용하여 센서 차원을 임베딩 차원으로 증강하고, 위치 인코딩과 값 인코딩을 추가합니다. 그 후, AnomalyAttention 클래스를 사용하여 쿼리, 키, 값, 시그마를 구성합니다. 이 값들은 훈련 과정에서 최적의 시리즈 연관성(series association)과 이전 연관성(prior association)을 구성하는 데 사용됩니다. 시리즈 연관성은 시간 지점 간의 유사성에 가중치를 부여하여 전체 시계열에 대한 정상 시간 지점의 유사성을 높입니다. Prior association은 인접한 시간 지점 간의 유사성에 가중치를 부여하여 이웃한 시간 지점의 정상 유사성을 고려합니다. 시리즈 연관성은 디코딩 레이어를 통해 원본 데이터로 복원됩니다. 훈련 과정에서는 입력값과 복원값, 시리즈 및 이전 연관성의 오차를 최소화하기 위해 훈련이 진행됩니다.

### 이상 탐지 및 시각화 (Anomaly detection and visualization)
훈련된 모델에 시험 데이터셋을 입력하면 각 데이터의 이상 점수를 얻을 수 있습니다. 각 시간 지점마다 출력된 이상 점수를 window size로 나누어 각 window의 가장 큰 값을 window의 대표 이상 점수로 설정합니다. 검증 데이터의 IQR 범위를 사용하여 임계값을 설정하고, 이상 데이터를 탐지합니다. 모델의 성능은 Confusion matrix와 F1-score를 사용하여 평가됩니다. Confusion matrix는 모델의 예측 값과 실제 값 사이의 관계를 나타내며, F1-score는 정밀도와 재현율의 조화 평균으로 이상 탐지 모델의 종합적인 성능을 평가합니다. Confusion matrix와 F1-score를 사용하여 모델의 성능을 직관적으로 파악할 수 있습니다.

## 요구 데이터 형식
코드 실행을 위해 데이터셋은 다음 format으로 저장되어야 합니다. 데이터셋은 CSV 파일로 구성되어 있으며, 각 파일은 센서별로 분리되어 있어야 합니다. 센서 데이터의 열(column)은 시간, 정상, 이상1, ... 이상n의 순서로 구성되어야 합니다. 입력값으로 주어진 폴더에는 다른 CSV 파일이 없어야 합니다. Preprocessing 함수를 사용하여 데이터를 모델에 적합한 형태로 변환할 수 있습니다.

## 참조
1. Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy(Jiehui Xu et al.)
2. KAMP-AI https://www.kamp-ai.kr/

## **사사문구(acknowledgement)**  
정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구임 (No.RS-2022-00155911, 인공지능융합혁신인재양성(경희대학교))     

Institute of Information &amp; communications Technology Planning &amp; Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2022-00155911, Artificial Intelligence Convergence Innovation Human Resources Development (Kyung Hee University))
