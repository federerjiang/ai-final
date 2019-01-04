import numpy as np
from env_args import EnvArgs 

import sys
sys.path.insert(0,'..')
import final_fixed_env as fixed_env
import load_trace as load_trace


class FixedEnvWrap(fixed_env.Environment):
	
	def __init__(self):
		self.args = EnvArgs()
		all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(self.args.test_bw_trace)
		super().__init__(all_cooked_time=all_cooked_time,
						 all_cooked_bw=all_cooked_bw,
						 random_seed=self.args.random_seed,
						 VIDEO_SIZE_FILE =self.args.test_video_size_files, 
						 logfile_path='./log/', 
						 Debug=False)
		
		self.state_gop = np.zeros((self.args.s_gop_info, self.args.s_gop_len)) # state info for past gops
		self.last_bit_rate = 0
		self.reward_gop = 0
		self.last_reward_gop = 0
		self.action_map = self._set_action_map()

		self.time_intervals = []
		self.send_data_sizes = []
		self.frame_types = []
		self.frame_time_lens = []
		self.real_qualitys = []
		self.buffer_sizes = []
		self.end_delays = []
		self.rebuf_time = 0

		self.call_time = 0
		self.switch_num = 0

		self.gop_sizes = [[0]*17, [0]*17]
		# info for traces
		self.traces_len = len(all_file_names)


	def _set_action_map(self):
		bit_rate_levels = [0, 1]
		target_buffer_levels = self.args.target_buffer
		action_map = []
		for bitrate_idx in range(len(bit_rate_levels)):
			for target_buffer_idx in range(len(target_buffer_levels)):
				action_map.append((bit_rate_levels[bitrate_idx], target_buffer_levels[target_buffer_idx]))
		return action_map


	def _update_state(self, cdn_flag, buffer_flag):
		block_size = np.array(self.send_data_sizes).sum() / 1000000

		# get average throughput in 0.5ms
		thps = []
		for time_interval, frame_size in zip(self.time_intervals, self.send_data_sizes):
			if time_interval > 0 and frame_size > 0:
				thps = frame_size / time_interval / 1000000
				break
		thp = np.array(thps).mean()

		# get gop sizes of 500k and 1200k
		for frame_tag, quality, frame_size in zip(self.frame_types, self.real_qualitys, self.send_data_sizes):
			if frame_tag == 1:
				self.gop_sizes[quality].append(frame_size)
				self.gop_sizes[quality].pop(0)
			else:
				self.gop_sizes[quality][-1] += frame_size

		# collect gop state info
		self.state_gop = np.roll(self.state_gop, -1, axis=1)
		self.state_gop[0, -1] = self.buffer_sizes[-1] # current buffer size [0, 10] [fc]
		self.state_gop[1, -1] = self.args.bitrate[self.last_bit_rate] / 1000 # last bitrate [0, 2] [fc]
		self.state_gop[2, -1] = thp # last throughput Mbps [0, 10] [conv]
		self.state_gop[3, -1] = block_size # last block sizes [conv]
		self.state_gop[4, -1] = self.frame_types.count(1) # record I frames count in one block. [conv] 
		self.state_gop[5, -1] = (1 if buffer_flag else 0) # if True, no buffering content, should choose target buffer as 0. [fc]
		self.state_gop[6, -1] = (1 if cdn_flag else 0) # if True, no content on cdn server. [fc]
		self.state_gop[7, :] = np.array(self.gop_sizes[0][:16]) / 1000000 # gop sizes of 500k [conv]
		self.state_gop[8, :] = np.array(self.gop_sizes[1][:16]) / 1000000 # gop sizes of 1200k [conv]

	# return gop state
	def step_gop(self, action):
		bit_rate, target_buffer = self.action_map[action]
		time, time_interval, send_data_size, frame_time_len, rebuf,\
		buffer_size, end_delay, cdn_newest_id,\
		download_id, cdn_has_frame, decision_flag, real_quality,\
		buffer_flag, switch, cdn_flag, end_of_video =\
		self.get_video_frame(bit_rate, target_buffer)

		self.time_intervals.append(time_interval)
		self.send_data_sizes.append(send_data_size)
		self.buffer_sizes.append(buffer_size)
		self.frame_time_lens.append(frame_time_len)
		self.end_delays.append(end_delay)
		self.real_qualitys.append(real_quality)
		if decision_flag:
			self.frame_types.append(1)
		else:
			self.frame_types.append(0)
		self.rebuf_time += rebuf

		self.call_time += time_interval
		self.switch_num += switch
		
		if not cdn_flag:
			reward_frame = frame_time_len * float(self.args.bitrate[bit_rate]) / 1000 \
							- self.args.rebuf_penalty * rebuf \
							- self.args.latency_penalty * end_delay
		else:
			reward_frame = -(self.args.rebuf_penalty * rebuf)

		if self.call_time > 0.5 and not end_of_video:
			assert self.switch_num <= 1
			reward_frame += -(self.switch_num) * self.args.smooth_penalty * 0.7

			self.last_bit_rate = self.real_qualitys[-1]
			self._update_state(cdn_flag, buffer_flag)

		self.reward_gop += reward_frame

		if self.call_time > 0.5 or end_of_video:
			self.last_reward_gop = self.reward_gop
			self.reward_gop = 0 # reset reward gop as 0
			self.call_time = 0
			self.switch_num = 0
			self.time_intervals = []
			self.send_data_sizes = []
			self.frame_types = []
			self.frame_time_lens = []
			self.real_qualitys = []
			self.buffer_sizes = []
			self.end_delays = []
			self.rebuf_time = 0
		
			return reward_frame, end_of_video, True
		else:
			return reward_frame, end_of_video, False


	def get_reward_gop(self):
		return self.last_reward_gop

	def get_state_gop(self):
		return self.state_gop

	def reset(self):
		self.state_gop = np.zeros((self.args.s_gop_info, self.args.s_gop_len))

		self.last_bit_rate = 0
		self.reward_gop = 0
		self.last_reward_gop = 0

		self.time_intervals = []
		self.send_data_sizes = []
		self.frame_types = []
		self.frame_time_lens = []
		self.real_qualitys = []
		self.buffer_sizes = []
		self.end_delays = []
		self.rebuf_time = 0

		self.call_time = 0
		self.switch_num = 0
		self.gop_sizes = [[0]*17, [0]*17]

		return self.state_gop

