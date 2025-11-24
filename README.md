# 강화학습에 광원&장애물 회피를 통한 구조 드론

[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=janghhh135)](https://github.com/anuraghazra/github-readme-stats)

<img width="900" height="314" alt="image" src="https://github.com/user-attachments/assets/c45af340-b7d1-4e7d-a7a4-0cc59f13373e" />

============================================================================================
# 광원(빛)의 위치를 탐지하고 해당 방향으로 이동하는 에이전트(agent)를 훈련시키는 것을 목표함.
- 광원 위치를 센서로 감지
- 에이전트는 관측값을 입력받아 행동 선택
- 강화학습을 통해 "빛에 가까워지는" 정책을 학습

# 강화학습 구조
✔ 사용 알고리즘 
- DQN

✔ Env
- Observation: 광원의 상대 위치
- Action: 좌/우 회전, 전진 속도, 회피 행동 등
- Reward:
  - 광원에 가까워지면 +reward
  - 멀어지면 -reward
  - 장애물 충돌 시 큰 penalty

============================================================================================
# 펌웨어
학습된 DQN 정책 모델을 불러옴
DQN → ONNX → TFLite → TFLite Micro → C 모델 → Crazyflie 펌웨어
다음과 같은 가정을 거쳐 crazyflie에 적용
