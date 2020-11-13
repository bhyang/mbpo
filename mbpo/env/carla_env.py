import os

from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import DEFAULT_ENV
import agents.tf.vis_module

import gym
from gym import utils

from datetime import datetime

class CarlaEnvWrapper(CarlaEnv, utils.EzPickle):
	def __init__(self, testing=False, seed=0):
		config = {}
		config['algo'] = 'MBPO'
		config['input_type'] = 'wp_obs_info_speed_steer_ldist_goal_light'
		config['action_type'] = 'merged_speed_scaled_tanh'
		config['scenarios'] = 'no_crash_dense'
		config['use_scenarios'] = True
		config['train_fixed_spawn_points'] = True
		config['test_fixed_spawn_points'] = True
		config['reward_function'] = 'simple2'
		config['sample_npc'] = True
		config['num_npc'] = 100
		config['city_name'] = 'Town01'
		config['const_collision_penalty'] = 150
		config['collision_penalty_speed_coeff'] = 150
		config['const_light_penalty'] = 150
		config['light_penalty_speed_coeff'] = 150
		config['vehicle_proximity_threshold'] = 10
		config['traffic_light_proximity_threshold'] = 10
		# config['steer_penalty_coeff'] = 2
		# config['constant_positive_reward'] = 1
		# config['success_reward'] = 75
		config['train_config'] = 'PPO'
		config['videos'] = testing
		config['testing'] = testing # testing
		config['frame_skip'] = 2
		config['reward_normalize_factor'] = 16
		# config['max_steps'] = 100
		config['verbose'] = False

		date = datetime.now().strftime('%Y-%m-%d')
		log_dir = '/home/brian/alta-logs/{}-seed={}'.format(date, seed)
		if not os.path.isdir(log_dir):
			os.mkdir(log_dir)

		IMAGES_PATH = '{}/test_images/'.format(log_dir)
		VIDEO_PATH = '{}/test_videos/'.format(log_dir)
		vis_wrapper = agents.tf.vis_module.vis(IMAGES_PATH, VIDEO_PATH, videos=testing)

		self._gym_disable_underscore_compat = True
		super(CarlaEnvWrapper, self).__init__(config=config, vis_wrapper=vis_wrapper, log_dir=log_dir)
		utils.EzPickle.__init__(self)