import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random




class BMSEnvironment(gym.Env):
    def __init__(self, **kwargs):
        super(BMSEnvironment, self).__init__()


        self.L = None
        self.G = None
        self.P = None
        self.SoC = None

        self.initial_SoC = kwargs.get('initial_SoC', 0.5)
        self.SoC_min = kwargs.get('SoC_min', 0.0)
        self.SoC_max = kwargs.get('SoC_max', 1.0)

        self.initial_load = kwargs.get('initial_load', 0.5)
        self.Load_min = kwargs.get('Load_min', 0.0)
        self.Load_max = kwargs.get('Load_max', 1.0)

        self.initial_generation = kwargs.get('initial_generation', 0.5)
        self.Generation_min = kwargs.get('Generation_min', 0.0)
        self.Generation_max = kwargs.get('Generation_max', 1.0)

        self.initial_price = kwargs.get('initial_price', 0.5)
        self.Price_min = kwargs.get('Price_min', 0.0)
        self.Price_mid = kwargs.get('Price_mid', 0.5)
        self.Price_max = kwargs.get('Price_max', 1.0)

        self.a_min = kwargs.get('a_min', -1.0)
        self.a_max = kwargs.get('a_max', 1.0)


        self.max_steps = kwargs.get('max_steps', 100)


        # time parameters
        self.time = kwargs.get('time', 0.0)
        self.hour_sin_min = kwargs.get('hour_sin_min', -1.0)
        self.hour_sin_max = kwargs.get('hour_sin_max', 1.0)
        self.hour_cos_min = kwargs.get('hour_cos_min', -1.0)
        self.hour_cos_max = kwargs.get('hour_cos_max', 1.0)
        self.hour_sin_initial = None
        self.hour_cos_initial = None

        self.day_sin_min = kwargs.get('day_sin_min', -1.0)
        self.day_sin_max = kwargs.get('day_sin_max', 1.0)
        self.day_cos_min = kwargs.get('day_cos_min', -1.0)
        self.day_cos_max = kwargs.get('day_cos_max', 1.0)
        self.day_sin_initial = None
        self.day_cos_initial = None
        # Penalties
        self.miu_p = kwargs.get('miu', 1.0)
        self.lamda_p = kwargs.get('lamda_', 1.0)


        # battery parameters
        self.eta = kwargs.get('eta', 0.9)
        self.battery_capacity = kwargs.get('battery_capacity', 1.0)

        self.time_interval = kwargs.get('time_interval', 1.0)  # h

        # information about the environment
        self.current_step = 0



        self.observation_space = spaces.Box(
            low=np.array([self.SoC_min, self.Load_min, self.Generation_min, self.hour_sin_min, self.hour_cos_min, self.day_sin_min, self.day_cos_min], dtype=np.float32),
            high=np.array([self.SoC_max, self.Load_max, self.Generation_max, self.hour_sin_max, self.hour_cos_max, self.day_sin_max, self.day_cos_max],dtype=np.float32))

        self.action_space = spaces.Box(
            low=np.array([self.a_min], dtype=np.float32),
            high=np.array([self.a_max], dtype=np.float32)
        )

    def _get_info(self):

        return {
            'SoC': self.SoC,
            'Load': self.L,
            'Generation': self.G,
            'Violations_SoC': [],
            'violations_action': [],
            'penalty_SoC': [],
            'penalty_action': []
        }

    def _get_obs(self):
        # time, hour encoding
        hour_sin, hour_cos = np.sin(2 * np.pi * self.time.hour / 24), np.cos(2 * np.pi * self.time.hour / 24)

        # day, day of week encoding
        day_sin, day_cos = np.sin(2 * np.pi * self.time.day / 7), np.cos(2 * np.pi * self.time.day / 7)

        observation = np.array([
            self.SoC,
            self.L,
            self.G,
            self.hour_sin,
            self.hour_cos,
            self.day_sin,
            self.day_cos
        ])

        return observation

    def _get_SoC(self, action, info=None):
        # Update SoC with the adjusted action
        SoC_proposed = self.SoC + self.eta * (action * self.time_interval) / self.battery_capacity

        # Check for SoC violations
        if SoC_proposed < self.SoC_min or SoC_proposed > self.SoC_max:
            info['violations_SoC'] = info.get('violations_SoC', [])
            info['violations_SoC'].append({
                'type': 'SoC Violation',
                'value': SoC_proposed,
                'corrected_to': np.clip(SoC_proposed, self.SoC_min, self.SoC_max)
            })

            # Clip SoC to stay within bounds
            SoC_adjusted = np.clip(SoC_proposed, self.SoC_min, self.SoC_max)
        else:
            SoC_adjusted = SoC_proposed

        # Calculate penalty for SoC adjustment
        delta_SoC = abs(SoC_proposed - SoC_adjusted)
        penalty_SoC = self.lamda_p * delta_SoC

        # Update the SoC
        self.SoC = SoC_adjusted

        # Store the penalty in info
        info['penalty_SoC'].append(penalty_SoC)

        return self.SoC, penalty_SoC

    def _get_action_check(self, action, info=None):

        adjusted_action = np.clip(action, self.action_space.low, self.action_space.high)
        delta_action = np.abs(action - adjusted_action)

        if action != adjusted_action:
            print(f'Action {action} is out of bounds. Adjusting to {adjusted_action}')
            info['violations_action'] = info.get('violations_action', [])
            info['violations_action'].append({
                'type': 'Action Violation',
                'value': action,
                'corrected_to': adjusted_action
            })

        penalty_action = self.miu_p * delta_action

        return adjusted_action, penalty_action

    def _get_price(self):

        if self.time.day <5:
            if  8 <= self.time.hour < 20:
                price = self.Price_max
            elif 7 <= self.time.hour < 8 or 20 <= self.time.hour < 23:
                price = self.Price_mid
            else:
                price = self.Price_min
        else:
            if 7 <= self.time.hour < 23:
                price = self.Price_mid
            else:
                price = self.Price_min

        return price



    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0

        self.SoC = self.initial_SoC
        self.G = self.initial_generation
        self.L = self.initial_load
        self.hour_sin = self.hour_sin_initial
        self.hour_cos = self.hour_cos_initial
        self.day_sin = self.day_sin_initial
        self.day_cos = self.day_cos_initial

        return self._get_obs(), self._get_info()

    def step(self,action):

        self.step += 1

        action_corrected, penalty_action = self._get_action_check(action, self._get_info())


        self.SoC, penalty_SoC = self._get_SoC(action_corrected, self._get_info())

        # Net energy
        P_grid = np.max(0, self.L - self.G + action_corrected)
        P_surplus = np.max(0, self.G - self.L - action_corrected)

        # Price
        price = self._get_price()

        C_purchased = np.min(P_grid, 0) * price

