import gymnasium as gym
from gymnasium import spaces
import numpy as np
import datetime

class BMSEnvironment(gym.Env):
    def __init__(self, **kwargs):
        super(BMSEnvironment, self).__init__()

        # State variables
        self.L = None  # Load demand
        self.G = None  # PV generation
        self.SoC = None  # State of Charge
        self.time = None  # Current time

        # Initial parameters
        self.initial_SoC = kwargs.get('initial_SoC', 0.5)
        self.SoC_min = kwargs.get('SoC_min', 0.1)
        self.SoC_max = kwargs.get('SoC_max', 0.95)

        self.Load_min = kwargs.get('Load_min', 0.0)
        self.Load_max = kwargs.get('Load_max', 1.0)

        self.Generation_min = kwargs.get('Generation_min', 0.0)
        self.Generation_max = kwargs.get('Generation_max', 1.0)

        self.Price_min = kwargs.get('Price_min', 0.1)
        self.Price_mid = kwargs.get('Price_mid', 0.2)
        self.Price_max = kwargs.get('Price_max', 0.3)

        self.a_min = kwargs.get('a_min', -1.0)  # Maximum discharging power
        self.a_max = kwargs.get('a_max', 1.0)   # Maximum charging power

        self.max_steps = kwargs.get('max_steps', 24)  # One day with hourly intervals

        # Penalties
        self.miu_p = kwargs.get('miu_p', 10.0)       # Penalty coefficient for action violations
        self.lamda_p = kwargs.get('lamda_p', 10.0)   # Penalty coefficient for SoC violations

        # Battery parameters
        self.eta = kwargs.get('eta', 0.9)  # Battery efficiency
        self.battery_capacity = kwargs.get('battery_capacity', 10.0)  # kWh

        self.time_interval = kwargs.get('time_interval', 1.0)  # h

        # Environment information
        self.current_step = 0

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=np.array([
                self.SoC_min, self.Load_min, self.Generation_min,
                -1.0, -1.0, -1.0, -1.0  # Time encodings (sin and cos values range from -1 to 1)
            ], dtype=np.float32),
            high=np.array([
                self.SoC_max, self.Load_max, self.Generation_max,
                1.0, 1.0, 1.0, 1.0
            ], dtype=np.float32)
        )

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
        # Time encoding
        hour = self.time.hour + self.time.minute / 60.0
        day_of_week = self.time.weekday()

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)

        observation = np.array([
            self.SoC,
            self.L,
            self.G,
            hour_sin,
            hour_cos,
            day_sin,
            day_cos
        ], dtype=np.float32)

        return observation

    def _get_SoC(self, action, info):
        # Update SoC with the adjusted action
        SoC_proposed = self.SoC + self.eta * (action * self.time_interval) / self.battery_capacity

        # Check for SoC violations
        if SoC_proposed < self.SoC_min or SoC_proposed > self.SoC_max:
            info['Violations_SoC'].append({
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

    def _get_action_check(self, action, info):
        adjusted_action = np.clip(action, self.action_space.low[0], self.action_space.high[0])
        delta_action = abs(action - adjusted_action)

        if action != adjusted_action:
            print(f'Action {action} is out of bounds. Adjusting to {adjusted_action}')
            info['violations_action'].append({
                'type': 'Action Violation',
                'value': action,
                'corrected_to': adjusted_action
            })

        penalty_action = self.miu_p * delta_action

        return adjusted_action, penalty_action

    def _get_price(self):
        day_of_week = self.time.weekday()  # 0 = Monday, ..., 6 = Sunday
        hour = self.time.hour

        if day_of_week < 5:  # Monday to Friday
            if 8 <= hour < 19:
                price = self.Price_max  # High price
            elif (7 <= hour < 8) or (19 <= hour < 23):
                price = self.Price_mid  # Medium price
            else:
                price = self.Price_min  # Low price
        elif day_of_week == 5:  # Saturday
            if 7 <= hour < 23:
                price = self.Price_mid  # Medium price
            else:
                price = self.Price_min  # Low price
        else:  # Sunday
            price = self.Price_min  # Low price

        return price

    def _update_generation_and_load(self):
        # Simulate PV generation (e.g., sinusoidal pattern to mimic daylight hours)
        hour = self.time.hour + self.time.minute / 60.0
        self.G = max(0, np.sin(np.pi * (hour - 6) / 12))  # Peak at noon

        # Simulate load demand (e.g., higher during morning and evening)
        self.L = 0.5 + 0.5 * np.sin(np.pi * (hour - 17) / 12)  # Peak around 5 PM

        # Normalize G and L to be within specified ranges
        self.G = np.clip(self.G, self.Generation_min, self.Generation_max)
        self.L = np.clip(self.L, self.Load_min, self.Load_max)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.SoC = self.initial_SoC
        self.time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Initialize PV generation and load demand
        self._update_generation_and_load()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        info = self._get_info()

        # Adjust the action and get any penalties
        action_corrected, penalty_action = self._get_action_check(action[0], info)

        # Enforce that the battery cannot charge and discharge at the same time
        # Since action_corrected is a scalar, and we have constraints ensuring it

        # Update SoC using the adjusted action
        self.SoC, penalty_SoC = self._get_SoC(action_corrected, info)

        # **Update PV Generation and Load Demand**
        self._update_generation_and_load()

        # **Energy Balance Calculations**

        # Net Load after PV generation (positive if demand exceeds generation)
        net_load = self.L - self.G

        # Determine battery action (charge/discharge/idle)
        # Ensure battery only discharges when there's net load to cover
        if action_corrected < 0:  # Discharging
            max_discharge = min(-action_corrected, net_load)
            actual_action = -max_discharge
        elif action_corrected > 0:  # Charging
            surplus_pv = max(0, -net_load)  # Surplus PV available
            max_charge = min(action_corrected, surplus_pv)
            actual_action = max_charge
        else:
            actual_action = 0.0  # Idle

        # Update SoC again with the actual action
        self.SoC, penalty_SoC_additional = self._get_SoC(actual_action - action_corrected, info)
        penalty_SoC += penalty_SoC_additional

        # Remaining net load after battery contribution
        net_load_after_battery = net_load + actual_action

        # Grid interaction
        if net_load_after_battery > 0:
            # Need to buy energy from the grid
            P_grid = net_load_after_battery
            P_surplus = 0.0
        elif net_load_after_battery < 0:
            # Surplus energy to sell to the grid
            P_grid = 0.0
            P_surplus = -net_load_after_battery
        else:
            P_grid = 0.0
            P_surplus = 0.0

        # Price
        price = self._get_price()

        # Costs and revenues
        C_purchased = price * P_grid
        R_sale = price * P_surplus  # Adjust if sell price differs

        # Total penalty
        total_penalty = penalty_action + penalty_SoC

        # Total reward
        reward = R_sale - C_purchased - total_penalty

        # Update time
        self.time += datetime.timedelta(hours=1)

        # Prepare observation
        observation = self._get_obs()

        # Check if episode is done
        done = self.current_step >= self.max_steps

        return observation, reward, done, False, info