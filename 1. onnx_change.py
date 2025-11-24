import torch
import torch.nn as nn
import os

# -----------------------------------------------------------
# 1. 모델 구조 (Q-Network) 정의 - 실제 모델 구조 [20, 20] 반영
# -----------------------------------------------------------
class SimpleQNetwork(nn.Module):
    # net_arch를 [20, 20]으로 수정
    def __init__(self, obs_dim, action_dim, net_arch=[20, 20], activation_fn=nn.ReLU):
        super().__init__()
        
        q_net = []
        
        # 0.weight: (obs_dim=6) -> (net_arch[0]=20)
        q_net.append(nn.Linear(obs_dim, net_arch[0])) 
        q_net.append(activation_fn())
        
        # 2.weight: (net_arch[0]=20) -> (net_arch[1]=20)
        q_net.append(nn.Linear(net_arch[0], net_arch[1]))
        q_net.append(activation_fn())
        
        # 4.weight: (net_arch[1]=20) -> (action_dim=3)
        q_net.append(nn.Linear(net_arch[1], action_dim))
        
        self.q_net = nn.Sequential(*q_net)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_net(obs)

# -----------------------------------------------------------
# 2. ONNX 변환 실행
# -----------------------------------------------------------

# 2-A. 모델 가중치 파일 경로 (로드할 파일 위치)
WEIGHT_PATH = r"C:\Users\jangh\Documents\pbl\sw_code\model\policy.pth"

# 2-B. ONNX 출력 파일 경로 (요청하신 C:\Users\jangh\Documents\pbl 폴더에 저장)
ONNX_OUTPUT_PATH = r"C:\Users\jangh\Documents\pbl\model.onnx"

# 2-C. 모델 메타데이터 정의 (오류 메시지를 바탕으로 실제 값으로 수정)
OBSERVATION_DIM = 6 # 관측값 차원 (입력)
ACTION_DIM = 3      # 동작 차원 (출력)

# 2-D. 모델 객체 생성 및 가중치 로드
model_to_export = SimpleQNetwork(OBSERVATION_DIM, ACTION_DIM)

# 'policy.pth' 파일에서 가중치 딕셔너리를 로드합니다.
state_dict = torch.load(WEIGHT_PATH, map_location=torch.device('cpu'))

# SB3 저장 형식(q_net.q_net.0.weight)을 SimpleQNetwork의 키에 맞게 변환
new_state_dict = {}
for k, v in state_dict.items():
    # 'q_net.q_net.' 접두사를 가진 일반 Q-Net 가중치만 추출
    if k.startswith('q_net.q_net.'): 
        new_k = k[12:] # 'q_net.q_net.' (12글자) 접두사 제거
        new_state_dict[new_k] = v
    # 만약 'q_net.'이 접두사라면
    elif k.startswith('q_net.'):
        new_k = k[6:] # 'q_net.' (6글자) 접두사 제거
        new_state_dict[new_k] = v

# 모델 객체에 가중치 적용
model_to_export.q_net.load_state_dict(new_state_dict) 
model_to_export.eval() # 추론 모드로 설정

# 2-E. ONNX 변환
dummy_input = torch.randn(1, OBSERVATION_DIM) 

torch.onnx.export(
    model_to_export, 
    dummy_input, 
    ONNX_OUTPUT_PATH, # <--- 지정된 경로
    opset_version=11, 
    export_params=True,
    input_names=['obs_input'], 
    output_names=['action_output']
)
