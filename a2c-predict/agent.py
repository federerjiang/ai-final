import torch
import torch.nn.functional as F 
import numpy as np
import random

from env_wrap import EnvWrap
from model import ActorCritic


def agent(rank, args, exp_queue, model_param):
	# env = EnvWrap(video_file_id, bw_trace='low')
	env = EnvWrap() 

	model = ActorCritic()
	model.load_state_dict(model_param.get())
	model.eval()

	state = env.reset()
	action = 3

	s_batch = [state]
	a_batch = [action]
	r_batch = []

	while True:
		reward_gop = 0
		while True:
			_, end_of_video, decision_flag = env.step_gop(action)
			if decision_flag or end_of_video:
				reward_gop = env.get_reward_gop()
				state = env.get_state_gop()
				break
			else:
				continue

		r_batch.append(reward_gop)

		with torch.no_grad():
			try:
				logit, value = model(torch.FloatTensor(state).view(-1, args.s_gop_info, args.s_gop_len))
				prob = F.softmax(logit, dim=1)
				action = prob.multinomial(1).data.numpy()[0][0]
			except RuntimeError:
				torch.save(state, 'state.pt')
				torch.save(logit, 'logit.pt')
				torch.save(prob, 'prob.pt')
				print('state: '. state)
				print('logit: ', logit)
				print('prob: ', prob)
				break

		done = end_of_video 

		if len(r_batch) >= args.max_update_step or done:
			if len(s_batch) >= 5:
				exp_queue.put([s_batch[1:],
								a_batch[1:],
								r_batch[1:],
								done])
				model.load_state_dict(model_param.get())
			del s_batch[:]
			del a_batch[:]
			del r_batch[:]
			# print('agent finish work')

		if done:
			state = env.reset()
			action = 3
			
		s_batch.append(state)
		a_batch.append(action)