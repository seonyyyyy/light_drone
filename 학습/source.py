import numpy as np

class LightSourceModel:
    def __init__(self, source_position, a=399.0, b=-2.6, c=5.1):
        self.source_position = source_position
        self.a = a
        self.b = b
        self.c = c
        self.noise_std = 4.0

    def get_light_intensity(self, drone_position):
        """
        드론 위치에 따른 빛의 세기를 노이즈를 포함하여 반환
        Vector3r 객체, 리스트, 튜플, numpy 배열 등 다양한 타입의 입력을 처리
        """
        # ----------------------------------------------------
        # --- 입력 타입 확인 및 변환 ---
        # ----------------------------------------------------
        pos_array = None
        # 입력이 .x_val 속성을 가진 객체인지 확인 (Vector3r 등)
        if hasattr(drone_position, 'x_val'):
            pos_array = np.array([drone_position.x_val, drone_position.y_val, drone_position.z_val])
        # 그렇지 않으면 리스트, 튜플, numpy 배열로 간주하고 변환
        else:
            pos_array = np.array(drone_position)

        # 변환 후 3개의 요소를 가졌는지 최종 확인
        if pos_array.shape != (3,):
            raise TypeError(f"Position data must have 3 elements, but got {drone_position}")
        # ----------------------------------------------------

        distance = np.linalg.norm(pos_array - self.source_position)
        intensity = self.a * np.exp(-((distance - self.b)**2) / (2 * self.c**2))
        noise = np.random.normal(0, self.noise_std)
        intensity += noise
        return intensity