import setup_path 
from gym import spaces
import airgym
import airsim
import time
import numpy as np
import os
import sys
import torch # ONNX 저장을 위해 torch 임포트

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# [수정] CheckpointCallback 임포트
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback 

# 프로젝트 루트 경로를 설정
sys.path.append(r"C:\Users\rlatn\Desktop\RL_Project\UAV_Airsim_RL_Project")
from airgym.envs.single_env import AirSimDroneEnv 

# --- 환경 생성 함수 ---
def create_env():
    return AirSimDroneEnv(ip_address="127.0.0.1", step_length=1.0)

# --- 로그 디렉토리 설정 ---
log_dir = "./single_log/"
os.makedirs(log_dir, exist_ok=True)

# [수정] 주기적 저장을 위한 체크포인트 디렉토리
checkpoint_dir = "./checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)

# --- 벡터 환경 구성 ---
env = make_vec_env(create_env, n_envs=1, monitor_dir=log_dir)

# *** 정책 네트워크 (기존과 동일) ***
policy_kwargs = dict(
    net_arch=[128, 128] 
)

# *** 하이퍼파라미터 (기존과 동일) ***
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.00025,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=20000, 
    learning_starts=50000,
    buffer_size=500000,
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log="./tb_logs/",
    policy_kwargs=policy_kwargs,
)

# --- [수정] 콜백 설정: EvalCallback + CheckpointCallback ---
callbacks = []

# 1. 평가 콜백 (기존과 동일)
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=log_dir, # 베스트 모델 저장 경로를 log_dir로 변경
    log_path=log_dir,            # 로그 경로를 log_dir로 변경
    eval_freq=10000,
)
callbacks.append(eval_callback)

# 2. [추가] 체크포인트 콜백
# 200만 스텝 학습이므로, 10만 스텝마다 저장 (총 20개 파일 생성)
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path=checkpoint_dir,
    name_prefix="dqn_drone_chk"
)
callbacks.append(checkpoint_callback)
# --- [수정 완료] ---


kwargs = {}
kwargs["callback"] = callbacks

# *** 학습 실행 (기존과 동일) ***
model.learn(
    total_timesteps=2e6, # 200만 스텝
    tb_log_name="dqn_drone_VER2_" + str(time.time()),
    **kwargs
)

# *** 최종 모델 저장 (기존과 동일) ***
model_save_path = "dqn_drone_policy_VER2_final" # 덮어쓰기 방지
model.save(model_save_path)
print(f"최종 모델을 {model_save_path}.zip 에 저장했습니다.")
env.close()