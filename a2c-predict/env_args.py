class EnvArgs:
	def __init__(self):
		# env settings
		self.s_gop_len = 16 # gop based info
		self.s_gop_info = 7 # or 7
		self.a_dim = 18
		self.random_seed = 10 # 50
		self.bitrate_levels = 2
		self.bitrate = [500.0, 1200.0] # kbps
		self.target_buffer_levels = 9
		self.target_buffer = [0.3, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4] # seconds

		self.frame_time_len = 0.04
		self.smooth_penalty = 0.02
		self.rebuf_penalty = 1.5
		self.latency_penalty = 0.005

		self.bw_trace = '../trace/network/final_network_trace/'
		self.test_bw_trace = '../trace/network/test/'
		self.video_size_files = '../trace/video/final_cooked_trace/'
		self.test_video_size_files = '../trace/video/test/'