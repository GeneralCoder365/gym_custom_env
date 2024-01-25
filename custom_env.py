import gym
from gym import spaces
from jax import numpy as jnp
import numpy as np
from jax import random

class CustomGridEnv(gym.Env):
    def __init__(self, grid_size=5, num_agents=2, num_adversaries=2, num_obstacles=2, seed=0):
        super(CustomGridEnv, self).__init__()
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_adversaries = num_adversaries
        self.num_obstacles = num_obstacles

        # Action: Four discrete actions (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Observation: Position of agents, adversaries, and obstacles
        self.observation_space = spaces.Box(low=0, high=self.grid_size, 
                                            shape=(self.num_agents + self.num_adversaries + self.num_obstacles, 2), 
                                            dtype=jnp.float32)

        self.state = None
        self.adversaries = None
        self.obstacles = None
        
        # Define target locations for each agent
        self.target_locations = self._initialize_target_locations()
        
        self.seed = seed
        self.key = random.PRNGKey(self.seed)

    def reset(self):
        # Initialize positions of agents, adversaries, and obstacles
        self.key, subkey = random.split(self.key)
        self.state = random.randint(subkey, shape=(self.num_agents, 2), minval=0, maxval=self.grid_size)

        self.key, subkey = random.split(self.key)
        self.adversaries = random.randint(subkey, shape=(self.num_adversaries, 2), minval=0, maxval=self.grid_size)

        self.key, subkey = random.split(self.key)
        self.obstacles = random.randint(subkey, shape=(self.num_obstacles, 2), minval=0, maxval=self.grid_size)

        return self._get_obs()

    def step(self, action):
        # Update positions of agents and adversaries
        new_state = self._update_positions(self.state, action[:self.num_agents])
        new_adversaries = self._update_positions(self.adversaries, action[self.num_agents:])

        # Check for collisions and update rewards
        reward = self._calculate_rewards(new_state, new_adversaries)
        done = [False] * (self.num_agents + self.num_adversaries)

        self.state = new_state
        self.adversaries = new_adversaries

        return self._get_obs(), reward, done, {}
    
    def _initialize_target_locations(self):
        # For simplicity, targets at fixed locations
        targets = np.array([(self.grid_size - 1, self.grid_size - 1) for _ in range(self.num_agents)])
        
        return targets

    def _get_target_locations(self):
        # Return the target locations
        return self.target_locations

    def _get_obs(self):
        # Combine state, adversaries, and obstacles into a single observation array
        return jnp.concatenate((self.state, self.adversaries, self.obstacles), axis=0)

    def _update_positions(self, positions, actions):
        # Movement deltas for actions (up, down, left, right)
        movement = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        new_positions = positions.copy()
        for i, action in enumerate(actions):
            action_int = action.item()  # Convert JAX array to integer
            if action_int not in movement:
                continue  # No action or invalid action

            # Calculate new position
            delta = movement[action_int]
            new_pos = (positions[i][0] + delta[0], positions[i][1] + delta[1])

            # Check boundary conditions and obstacles
            if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                if not any(jnp.array_equal(new_pos, obs) for obs in self.obstacles):
                    new_positions = new_positions.at[i].set(new_pos)

        return new_positions

    def _calculate_rewards(self, new_state, new_adversaries):
        # Initialize rewards
        rewards = jnp.zeros(self.num_agents + self.num_adversaries)

        # Rewards for agents
        for i in range(self.num_agents):
            distance_to_target = jnp.linalg.norm(new_state[i] - self.target_locations[i])
            rewards = rewards.at[i].set(-distance_to_target)  # Negative reward for distance

        # Rewards for adversaries
        for i, adv_position in enumerate(new_adversaries):
            closest_agent_dist = min(jnp.linalg.norm(adv_position - agent_position) for agent_position in new_state)
            idx = self.num_agents + i
            rewards = rewards.at[idx].set(-closest_agent_dist)  # Update rewards for adversaries

        return rewards