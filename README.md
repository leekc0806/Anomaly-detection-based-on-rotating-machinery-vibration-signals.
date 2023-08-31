# 회전기계 진동 신호 기반 이상탐지
이 프로젝트는 회전기계의 진동 데이터에 기반하여 이상 탐지 모델링, 평가 기술을 제공합니다. 회전기계는 산업 분야에서 중요한 장비로 사용되며, 이들의 안정적인 운전을 위해 진동 데이터 모니터링은 필수적입니다. 프로젝트의 코드를 통해 다양한 센서에서 수집된 다변량 시계열 데이터를 활용하여 회전기계의 상태를 실시간으로 평가하고, 이상 징후를 탐지하는데 도움을 줍니다.

## 기능
코드의 데이터 전처리 과정은 데이터를 효율적으로 처리하고 모델 학습에 적합하도록 준비합니다. 선형 보간법을 사용하여 데이터의 Sampling rate를 맞춰주고, 센서 단위로 구성된 데이터를 클래스 별로 분류합니다. 이렇게 구성된 데이터는 모델 학습을 위해 변환됩니다. 이 때, 이상 데이터는 현실에서 드물게 발생하기 때문에 모델을 훈련할 때는 정상 데이터만 사용합니다.

인공지능 모델로 Anomaly transformer라는 이상 탐지 모델을 채택합니다. 이 모델은 각 시간 지점 간의 상관관계를 분석하여 이상이 발생하는 시간 지점을 검출합니다. 이를 위해 전처리된 훈련 데이터를 사용하여 모델을 학습하고, 검증 데이터를 활용하여 훈련 데이터에 과적합되었는지 확인합니다. 또한, Early stopping을 통해 적절한 학습 반복 횟수를 결정합니다.

테스트 단계에서는 훈련된 모델에 테스트 데이터를 입력하여 각 윈도우 별로 이상 점수를 계산합니다. F1-score를 통해 모델의 성능을 평가하고 Confusion matrix를 통해서 시각화합니다. 이를 통해 모델의 성능을 직관적으로 알 수 있습니다. 본 프로그램은 특정 데이터셋에 국한되지 않고, 사용 방법을 따라하면 다른 다변량 시계열 데이터셋에도 적용할 수 있도록 설계되었습니다.

## 참조
Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy(Jiehui Xu et al.)
KAMP-AI https://www.kamp-ai.kr/

## **Ackowledgement**  
정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구임 (No.RS-2022-00155911, 인공지능융합혁신인재양성(경희대학교))     

Institute of Information &amp; communications Technology Planning &amp; Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2022-00155911, Artificial Intelligence Convergence Innovation Human Resources Development (Kyung Hee University))
