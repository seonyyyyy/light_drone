import setup_path 
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
import sys

from airgym.envs.airsim_env import AirSimEnv
# from airgym.envs.source import LightSourceModel # 광원 모델 임포트 제거

class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length):
        super().__init__()
        self.step_length = step_length
        
        # [수정] 보상 계산에 필요한 변수 추가
        self.state = {
            # airsim.Vector3r 객체를 저장하도록 변경 (np.zeros(3) 대신)
            "position": airsim.Vector3r(), 
            "collision": False,
            "prev_position": airsim.Vector3r(),
            "laser_rangers": np.zeros(4, dtype=np.float32), # 라이다 값 저장
            "prev_dist_to_target": 0.0  # ◀◀◀ [핵심 추가] 이전 스텝의 2D 거리
        }
        
        # [수정] 관측 공간 (기존과 동일)
        low = np.array([-100.0, -100.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([100.0, 100.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(6,), dtype=np.float32)

        self.drone = airsim.MultirotorClient(ip=ip_address)
        
        # [수정] 액션 스페이스 (기존과 동일)
        self.action_space = spaces.Discrete(3)
        self.agent_start_pos = np.array([-5, 0, -12]) # 고정된 시작 위치
        self.target_position = self.agent_start_pos  # reset에서 덮어쓸 임시값
        
        # --- [핵심 추가] 보상 하이퍼파라미터 및 환경 설정 ---
        self.debug = True  # ◀◀◀ 디버깅 모드 플래그
        self.LIDAR_MAX_DIST = 5.0
        
        # [수정 1] 성공/실패 보상의 절대값을 2배로 늘림
        self.R_GOAL = 1000.0                # 🌟 (500 -> 1000)
        self.R_CRASH = -1000.0              # 💥 (500 -> 1000)
        
        # [수정 2] "엔진" 보상(R_dist) 강화
        self.K_DISTANCE = 300.0             # 🎯 (200 -> 300)
        
        # [수정 3] "브레이크" 페널티(R_prox) 완화
        self.R_PROXIMITY_PENALTY = -5.0     # ⚠️ (-10 또는 -5 -> -5 유지)
        
        # [수정 4] "시간" 페널티 완화
        self.R_TIME = -0.1                  # ⏳ (-0.5 -> -0.1)

        # [수정 5] "브레이크 민감도" 완화
        self.GOAL_THRESHOLD_2D = 5.0
        self.DANGER_THRESHOLD = 0.3         # ◀ (0.5 -> 0.3) 0.3m까지는 봐줌
        
        # [수정 6] "포기" 임계값 (Geofence)
        self.TOO_FAR_THRESHOLD = 60.0       # (60m 유지)
        
        # [수정 7] "포기" 페널티를 "충돌"보다 더 나쁘게 설정 (핵심)
        self.R_TOO_FAR_PENALTY = -1200.0    # 🚫 (-200 -> -1200)

        # 에피소드 통계
        self.max_steps = 800 # (기존 코드)
        self.current_step = 0
        self.current_episode_reward = 0.0 
        self.episode_count = 0
        
        # (중요) `__init__`의 마지막에서 비행 설정을 호출해야 합니다.
        self._setup_flight()


    def close(self):
        super().close()
        try:
            if self.drone:
                self.drone.reset()
                self.drone.enableApiControl(False, vehicle_name="Drone1")
                self.drone.enableApiControl(False, vehicle_name="Drone2")
        except Exception as e:
            print(f"Error during AirSimDroneEnv close: {e}")

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        # ▼▼▼ [수정] np.int64를 float()로 변환 ▼▼▼
        self.drone.moveToPositionAsync(
            float(self.agent_start_pos[0]), 
            float(self.agent_start_pos[1]), 
            float(self.agent_start_pos[2]), 
            10
        ).join()

        self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()

    # *** 변경: 함수명 및 로직 수정 (Light -> Target)
    def _randomize_target_position(self):
        """
        [수정] 에이전트 시작 위치를 기준으로 반경을 설정합니다.
        (처음 4번 에피소드는 5m 이내, 그 이후는 5-50m)
        """
        # 1. 랜덤 각도 생성 (0 ~ 360도)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # --- [핵심 수정] 에피소드 카운트에 따라 반경(radius) 조절 ---
        # (가정) self.episode_count는 reset()에서 1부터 시작하며, 이 함수 호출 전에 증가됨.
        if self.episode_count <= 4:
            # 초반 4개 에피소드는 1m ~ 5m 사이의 가까운 거리
            # (0m가 아닌 1m부터 시작하여 타겟과 겹치지 않게 함)
            radius = np.random.uniform(3.0, 7.0) 
        else:
            # 그 이후 에피소드는 5m ~ 50m 사이의 먼 거리 (기존 로직)
            radius = np.random.uniform(20.0, 50.0)
        # --- [수정 완료] ---

        # 3. 직교 좌표계로 변환 (X, Y)
        offset_x = radius * np.cos(angle)
        offset_y = radius * np.sin(angle)
        
        # 4. 새로운 타겟 위치 계산
        new_target_pos = [
            self.agent_start_pos[0] + offset_x,
            self.agent_start_pos[1] + offset_y,
            self.agent_start_pos[2]  # 고도는 시작 고정과 동일
        ]
        
        # self.target_position 업데이트
        self.target_position = np.array(new_target_pos)
        
        # [수정] 디버깅 print문으로 현재 반경과 에피소드 번호 표시
        print(f"[Ep {self.episode_count}] New target position: [{new_target_pos[0]:.1f}, {new_target_pos[1]:.1f}] (Radius: {radius:.1f}m)")


        # --- 추가된 부분: 깃발 객체(목표지) 이동 ---
        try:
            object_name = "target1v1_2" 

            flag_position = airsim.Vector3r(
                float(new_target_pos[0]), 
                float(new_target_pos[1]), 
                float(new_target_pos[2])
            )
            
            flag_orientation = airsim.to_quaternion(0, 80.1, 0) # 깃발 방향 (필요시 조절)
            flag_pose = airsim.Pose(flag_position, flag_orientation)
            
            self.drone.simSetObjectPose(object_name, flag_pose)

        except Exception as e:
            print(f"'{object_name}' 객체를 이동 실패. 언리얼 레벨에 해당 이름의 객체가 있는지 확인.")
            print(e)
    '''
    --------------------------------------------------------------------------------------------------------
    굳이 필요 없을거 같아서 주석처리 해둠 
    --------------------------------------------------------------------------------------------------------
    def transform_obs(self, responses):
        ... (이하 동일) ...
    '''
    
    def rotate_vector(self, vec, q):
        # q: airsim.Quaternionr
        # vec: airsim.Vector3r
        w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
        # 쿼터니언 회전 행렬
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ])
        v = np.array([vec.x_val, vec.y_val, vec.z_val])
        v_rot = R @ v
        return airsim.Vector3r(*v_rot)

    def _get_obs(self):
        self.drone_state = self.drone.getMultirotorState()
        
        # [수정] airsim.Vector3r 객체로 저장
        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        self.state["collision"] = self.drone.simGetCollisionInfo().has_collided
        
        # 1. 4방향 라이다 센서값
        front_dist = self.drone.getDistanceSensorData(distance_sensor_name="FrontDistance").distance
        back_dist = self.drone.getDistanceSensorData(distance_sensor_name="BackDistance").distance
        left_dist = self.drone.getDistanceSensorData(distance_sensor_name="LeftDistance").distance
        right_dist = self.drone.getDistanceSensorData(distance_sensor_name="RightDistance").distance
        
        # [수정] self.LIDAR_MAX_DIST 사용
        self.state["laser_rangers"] = np.array([
            min(front_dist, self.LIDAR_MAX_DIST),
            min(right_dist, self.LIDAR_MAX_DIST),
            min(back_dist, self.LIDAR_MAX_DIST),
            min(left_dist, self.LIDAR_MAX_DIST)
        ], dtype=np.float32)
        
        # 2. 상대 벡터 계산 (Body Frame)
        current_pos_np = np.array([
            self.state["position"].x_val,
            self.state["position"].y_val,
            self.state["position"].z_val
        ])
        relative_vector_3d_world = self.target_position - current_pos_np
        
        orientation_q = self.drone_state.kinematics_estimated.orientation
        yaw_rad = airsim.to_eularian_angles(orientation_q)[2]
        
        cos_yaw = np.cos(-yaw_rad)
        sin_yaw = np.sin(-yaw_rad)
        
        world_x = relative_vector_3d_world[0]
        world_y = relative_vector_3d_world[1]
        
        body_x = world_x * cos_yaw - world_y * sin_yaw
        body_y = world_x * sin_yaw + world_y * cos_yaw
        
        relative_vector_2d_body = np.array([body_x, body_y], dtype=np.float32)
        
        # 3. [수정] 관측값 순서 변경: [상대벡터(2), 라이다(4)]
        observation = np.concatenate([
            relative_vector_2d_body, 
            self.state["laser_rangers"]
        ]).astype(np.float32)
        
        return observation

    def _do_action(self, action):
        # 이동 및 회전 속도 설정
        speed = 1  # 초당 1미터 속도로 전진
        yaw_rate = 15 # 초당 15도 속도로 회전
        duration = 0.5 # 각 액션을 0.5초 동안 지속

        # 0: 앞으로 이동
        if action == 0:
            orientation = self.drone.getMultirotorState().kinematics_estimated.orientation
            forward_vec = airsim.Vector3r(1, 0, 0)
            rotated_forward = self.rotate_vector(forward_vec, orientation)

            vx = rotated_forward.x_val * speed
            vy = rotated_forward.y_val * speed
            vz = rotated_forward.z_val * speed 

            self.drone.moveByVelocityAsync(vx, vy, vz, duration).join()

        # 1: 왼쪽으로 회전
        elif action == 1:
            self.drone.rotateByYawRateAsync(-yaw_rate, duration).join()
            
        # 2: 오른쪽으로 회전
        elif action == 2:
            self.drone.rotateByYawRateAsync(yaw_rate, duration).join()

    def _compute_reward(self):        
        # --- 0. 디버깅 변수 초기화 ---
        r_time, r_crash, r_goal, r_prox, r_dist, r_far = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        done = False

        # --- 1. 시간 페널티 (R_time) ---
        r_time = self.R_TIME

        # --- 2. 충돌 페널티 (R_crash) ---
        if self.state["collision"]:
            r_crash = self.R_CRASH
            done = True
            total_reward = r_time + r_crash
            if self.debug: 
                print(f"DEBUG: 💥 CRASHED! | Total Reward: {total_reward:.2f}")
            return total_reward, done

        # --- 3. 2D 거리 계산 ---       
        current_pos_2d = np.array([
            self.state["position"].x_val,
            self.state["position"].y_val
        ])
        target_pos_2d = self.target_position[:2]
        
        # 0으로 나누기 방지를 위해 epsilon 추가
        current_dist_to_target_2D = np.linalg.norm(current_pos_2d - target_pos_2d) + 1e-6

        # --- 4. 목표 도달 보상 (R_goal) ---
        # [수정] 2D 거리 2.0m 이내인지 확인 (self.GOAL_THRESHOLD_2D 사용)
        if current_dist_to_target_2D <= self.GOAL_THRESHOLD_2D:
            r_goal = self.R_GOAL
            done = True
            total_reward = r_time + r_goal
            if self.debug: 
                print(f"DEBUG: 🌟 GOAL REACHED! (2D Dist: {current_dist_to_target_2D:.2f}m) | Total Reward: {total_reward:.2f}")
            return total_reward, done
        
        # --- 5."너무 멀어짐" 페널티 (터미널) ---
        if current_dist_to_target_2D > self.TOO_FAR_THRESHOLD:
            r_far = self.R_TOO_FAR_PENALTY
            done = True
            # (중요) 다른 보상(r_dist, r_prox)은 0인 상태로 종료
            total_reward = r_time + r_far 
            if self.debug:
                print(f"DEBUG: 🚫 TOO FAR! (Dist: {current_dist_to_target_2D:.1f}m) | Total Reward: {total_reward:.2f}")
            return total_reward, done
        
        # --- 6. 장애물 근접 페널티 (R_proximity) ---
        # [수정] 가우시안 대신 단순 임계값 페널티로 변경
        min_laser_dist = np.min(self.state["laser_rangers"])
        
        if min_laser_dist < self.DANGER_THRESHOLD:
            # r_prox = self.R_PROXIMITY_PENALTY
            r_prox = self.R_PROXIMITY_PENALTY * (self.DANGER_THRESHOLD - min_laser_dist) / self.DANGER_THRESHOLD

        # --- 7. 거리 기반 보상 (R_distance) ---        
        progress = self.state["prev_dist_to_target"] - current_dist_to_target_2D
        r_dist = progress * self.K_DISTANCE

        # (필수) 다음 스텝을 위해 "이전 거리" 값을 현재 2D 거리로 업데이트
        self.state["prev_dist_to_target"] = current_dist_to_target_2D
        
        # --- 7. 최종 보상 합산 및 디버깅 출력 ---
        total_reward = r_time + r_goal + r_crash + r_prox + r_dist + r_far

        # [수정] 디버깅 프린트 포맷 변경
        if self.debug:
            print(f"  [REWARD] Total: {total_reward: >8.2f} | "
                  f"R_dist(P): {r_dist: >7.2f} (Prog: {progress: >+5.2f}m) | "
                  f"R_prox: {r_prox: >6.1f} (Safe: {min_laser_dist: >4.1f}m) | "
                  f"R_time: {r_time: >4.1f} | "
                  f"Dist(2D): {current_dist_to_target_2D: >5.1f}m")

        # [수정] done 플래그만 리턴 (기존 alpha, beta 대신)
        return total_reward, done


    def step(self, action):
        self.current_step += 1

        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward() # ◀ (alpha, beta 대신 done을 직접 받음)

        # 최대 스텝 수에 도달했는지 확인
        max_step_reached = self.current_step >= self.max_steps
        if max_step_reached:
            done = True # ◀ max_step 도달 시 강제 종료
            if self.debug: print(f"DEBUG: 🕖 Max steps ({self.max_steps}) reached.")

        info = {}
        self.current_episode_reward += reward
        
        # info 딕셔너리에 타임아웃 여부 추가 
        if done and max_step_reached:
            info['TimeLimit.truncated'] = True

        return obs, reward, done, info


    def reset(self):
        # --- (1) 기존 에피소드 요약 출력 ---
        if self.current_step > 0:
            print("*" * 30)
            print(f"EPISODE {self.episode_count} FINISHED") # ◀ 에피소드 카운트 표시
            print(f"Total Reward: {self.current_episode_reward:.2f}")
            print(f"Total Steps:  {self.current_step}")
            print("*" * 30)

        # --- (2) 카운터 리셋 ---
        self.current_step = 0
        self.current_episode_reward = 0.0
        self.episode_count += 1
        
        # --- (3) 환경 리셋 (타겟 및 드론 위치) ---
        self._randomize_target_position() # 1. self.target_position 설정
        self._setup_flight()              # 2. 드론 시뮬레이션 위치 리셋
        
        # --- (4) 초기 관측값 획득 ---
        # (중요) _get_obs()가 self.state["position"]을 초기 위치로 업데이트함
        observation = self._get_obs() 
        
        # --- (5) [필수 수정] Progress 보상을 위한 초기 거리 계산 ---
        
        # _get_obs()가 방금 업데이트한 초기 드론 위치 (2D)
        initial_pos_np = np.array([
            self.state["position"].x_val,
            self.state["position"].y_val
        ])
        # _randomize_target_position()이 설정한 타겟 위치 (2D)
        target_pt_2d = self.target_position[:2]
        
        # 초기 2D 거리 계산 (0으로 나누기 방지)
        initial_dist_2d = np.linalg.norm(initial_pos_np - target_pt_2d) + 1e-6
        
        # [핵심] "prev_dist_to_target" 값을 현재 초기 거리로 설정
        self.state["prev_dist_to_target"] = initial_dist_2d
        # --- [수정 완료] ---

        # --- (6) 디버깅 출력 ---
        if self.debug: 
            print(f"\n====== EPISODE {self.episode_count} RESET ======")
            print(f"  New Target (2D): [{target_pt_2d[0]:.1f}, {target_pt_2d[1]:.1f}]")
            print(f"  Initial 2D Dist: {initial_dist_2d:.2f}m")

        # --- (7) 초기 관측값 반환 ---
        return observation