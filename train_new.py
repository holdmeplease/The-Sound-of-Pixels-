from models import modifyresnet18, UNet, Synthesizer
import torch
import torch.optim as optim
import torch.nn as nn
from util.datahelper import load_all_training_data, sample_from_dict #正确性未经验证
import itertools
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os
from torch.autograd import Variable
from util.validation import spec2wave,compute_sdr
from librosa import amplitude_to_db

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def data_in_one(inputdata): #将数据映射到0-1之间
	inputdata = (inputdata-inputdata.min())/(inputdata.max()-inputdata.min())
	return inputdata

def train(spec_dir, image_dir, model_dir, batch_size=8, validate_freq=200):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	video_net = modifyresnet18(batch_size).to(device) #载入网络
	audio_net = UNet().to(device)
	audio_net._initialize_weights()
	syn_net = Synthesizer().to(device)
	syn_net._initialize_weights()
	#print('gpu',device)
	if os.path.exists(os.path.join(model_dir, 'video_net_params.pkl')) and os.path.exists(
		os.path.join(model_dir, 'audio_net_params.pkl')) and os.path.exists(
		os.path.join(model_dir, 'syn_net_params.pkl')):
		print('load params!')
		video_net.load_state_dict(torch.load(os.path.join(model_dir, 'video_net_params.pkl')))
		audio_net.load_state_dict(torch.load(os.path.join(model_dir, 'audio_net_params.pkl')))
		syn_net.load_state_dict(torch.load(os.path.join(model_dir, 'syn_net_params.pkl')))
	video_net.train()
	audio_net.train()
	syn_net.train()
	optim_video = optim.SGD(video_net.parameters(), lr=0.0001, momentum=0.9) #定义优化器
	optim_audio = optim.SGD(audio_net.parameters(), lr=0.001, momentum=0.9)
	optim_syn = optim.SGD(syn_net.parameters(), lr=0.001, momentum=0.9)
	#optim_video.zero_grad() #初始化参数梯度
	#optim_audio.zero_grad()
	#optim_syn.zero_grad()
		
	[spec_data, image_data] = load_all_training_data(spec_dir, image_dir) #载入训练数据
	#print('spec_data_len',len(spec_data))
	#print('image_data_len',len(image_data))
	#print('spec_data',spec_data)
	#print('image_data',image_data)
	print('data loaded')
	criterion = nn.L1Loss() #binary cross entropy
	running_loss = 0.0 
	count = 0
	for num_batch in itertools.count(): #训练
		if num_batch > 20000:
			break
		spec_input_sampled = np.zeros((2, batch_size, 256, 256), dtype='complex') 
		image_input_sampled = np.zeros((2, 3 * batch_size, 3, 224, 224), dtype='float32') 
		for i in range(batch_size):
			[spec_input_mini, image_input_mini] = sample_from_dict(spec_data, image_data) 
			spec_input_sampled[:, i:i+1, :, :] = spec_input_mini #(2,batch_size,256,256)
			image_input_sampled[:, 3*i:3*i+3, :, :, :] = image_input_mini #(2,3*batch_size,3,224,224)
		#print('spec_input_sampled',spec_input_sampled.shape)
		#print('image_input_sampled',image_input_sampled.shape) 
		spec_input1 = np.transpose(np.reshape(spec_input_sampled[0,:,:,:], (1,batch_size,256,256)), (1,0,2,3))#第一种乐器频谱
		spec_input2 = np.transpose(np.reshape(spec_input_sampled[1,:,:,:], (1,batch_size,256,256)), (1,0,2,3)) #第二种乐器频谱
		spec_input = spec_input1 + spec_input2 #总频谱
		#print('spec_input',spec_input.shape)
		spec_abs1 = np.absolute(spec_input1)
		spec_abs2 = np.absolute(spec_input2)
		spec_abs = spec_abs1 + spec_abs2
		#spec_abs = np.absolute(spec_input) #总幅度谱
		mask1_ratio = np.zeros((batch_size, 1, 256, 256), dtype='float32')
		mask2_ratio = np.zeros((batch_size, 1, 256, 256), dtype='float32')
		for idx_0 in range(batch_size):
			for idx_2 in range(256):
				for idx_3 in range(256):
					if spec_abs[idx_0,0,idx_2,idx_3] == 0:
						mask1_ratio[idx_0,0,idx_2,idx_3] = 0.5
						mask2_ratio[idx_0,0,idx_2,idx_3] = 0.5
					else:
						mask1_ratio[idx_0,0,idx_2,idx_3] = spec_abs1[idx_0,0,idx_2,idx_3]/spec_abs[idx_0,0,idx_2,idx_3]
						mask2_ratio[idx_0,0,idx_2,idx_3] = spec_abs2[idx_0,0,idx_2,idx_3]/spec_abs[idx_0,0,idx_2,idx_3]
		#mask1_ratio = np.absolute(spec_input1)/spec_abs
		#print('mask1_ratio',mask1_ratio)
		#print('mask2_ratio',mask2_ratio)
		#mask2_ratio = np.absolute(spec_input2)/spec_abs
		spec_dp = torch.from_numpy(amplitude_to_db(spec_abs)).float().to(device) #总幅度谱取dp
		#print('spec_abs',spec_abs.shape)
		#mask_input2 = np.transpose(np.reshape(np.argmax(spec_input_sampled, axis=0),(1,batch_size,256,256)), (1,0,2,3)) #计算mask
		#mask_input1 = np.ones((batch_size,1,256,256))-mask_input2
		#mask_input1 = torch.from_numpy(mask_input1).float().to(device) #mask1
		#mask_input2 = torch.from_numpy(mask_input2).float().to(device) #mask2
		#print('mask_input',mask_input2.shape)
		#print('mask_input',mask_input1)
		optim_video.zero_grad() #初始化参数梯度
		optim_audio.zero_grad()
		optim_syn.zero_grad()
		
		out_audio_net = audio_net(spec_dp) #数据通路
		#print('out_audio_net',out_audio_net.shape)
		image_input1 = torch.from_numpy(image_input_sampled[0,:,:,:,:]).float().to(device) #video1
		#print('image_input_type',image_input1.type)
		#print('image_input_grad',image_input1.grad)
		#print('image_input',image_input1.shape)
		image_input2 = torch.from_numpy(image_input_sampled[1,:,:,:,:]).float().to(device)
		#image_input2 = (torch.from_numpy(image_input_sampled[1,:,:,:,:])).float().to(device) #video2
		out1_video_net = video_net(image_input1) #送进video网络
		#print('out_video_net',out1_video_net.shape)
		out2_video_net = video_net(image_input2)
		input1_syn_net = out1_video_net * out_audio_net #送进syn网络之前
		#print('input_syn_net_0',input1_syn_net.shape)
		input2_syn_net = out2_video_net * out_audio_net
		input1_syn_net = torch.transpose(input1_syn_net,1,2) #转置以匹配维度
		input1_syn_net = torch.transpose(input1_syn_net,2,3)
		#print('input_syn_net_1',input1_syn_net.shape)
		input2_syn_net = torch.transpose(input2_syn_net,1,2)
		input2_syn_net = torch.transpose(input2_syn_net,2,3)
		out1_syn_net = syn_net(input1_syn_net)
		#print('out_syn_net_0',out1_syn_net.shape)
		out2_syn_net = syn_net(input2_syn_net)
		out1_syn_net = torch.transpose(out1_syn_net,2,3) #转置以匹配维度
		out1_syn_net = torch.transpose(out1_syn_net,1,2)
		#print('out_syn_net_1',out1_syn_net.shape)
		#print(out1_syn_net)
		out2_syn_net = torch.transpose(out2_syn_net,2,3)
		out2_syn_net = torch.transpose(out2_syn_net,1,2) #(batch_size,1,256,256)
		#print('out1_syn_net.shape',out1_syn_net)
		#out1_syn_net_binary = torch.round(out1_syn_net)
		#out2_syn_net_binary = torch.round(out2_syn_net)
		#print('out1_syn_net_binary',out1_syn_net_binary)
		#s1_estimated = out1_syn_net * torch.from_numpy(spec_abs).float().to(device) #幅度谱估计
		#print('s_estimated',s1_estimated.shape)
		#s2_estimated = out2_syn_net * torch.from_numpy(spec_abs).float().to(device)
		#s1_estimated_np = s1_estimated.detach().numpy()
		#librosa.display.specshow(librosa.amplitude_to_db(np.abs(spec_input1[0,:,:]),ref=np.max),y_axis='log', x_axis='time') #画幅度谱
		#print('done!')
		#print(np.shape(spec_input1))
		#plt.title('Power spectrogram')
		#plt.colorbar(format='%+2.0f dB')
		#plt.tight_layout()
		#plt.show()
		#librosa.display.specshow(librosa.amplitude_to_db(np.abs(s1_estimated_np[0,0,:,:]),ref=np.max),y_axis='log', x_axis='time')
		#print('done!')
		#print(np.shape(spec_input1))
		#plt.title('Power spectrogram')
		#plt.colorbar(format='%+2.0f dB')
		#plt.tight_layout()
		#plt.show()
		#损失函数，可能有问题
		loss1 = criterion(out1_syn_net, torch.from_numpy(mask1_ratio).float().to(device))
		loss2 = criterion(out2_syn_net, torch.from_numpy(mask2_ratio).float().to(device))
		loss = loss1 + loss2
		#out_audio_net.register_hook(print)
		#out1_video_net.register_hook(print)
		#out2_syn_net.register_hook(print)
		#input1_syn_net.register_hook(print)
		#loss1.register_hook(print)
		#loss.register_hook(print)
		loss.backward() #反向传播
		#print('image_input_grad',image_input1.grad)
		#print('out1_video_net_grad',out1_video_net.grad)
		#print('loss1_type',loss1.type)
		#print('loss2_type',loss2.type)
		#print('loss_type',loss.type)
		#print('loss1_grad',loss1.requires_grad)
		#print('loss2_grad',loss2.requires_grad)
		#print('loss_grad',loss.requires_grad)
		#print('out2_syn_net',out2_syn_net.requires_grad)
		#print('input1_syn_net',input1_syn_net.requires_grad)
		#print('out2_video_net',out2_video_net.requires_grad)
		#print('out_audio_net',out_audio_net.requires_grad)
		
		optim_syn.step()
		optim_audio.step()
		optim_video.step() #迭代一步
		#print('loss_backward',loss.backward)
		running_loss += loss.item() #打印loss
		if num_batch % 200 == 199:
			print('[%5d] loss: %.5f' % (num_batch+1, running_loss/200))
			running_loss = 0.0
		if num_batch % validate_freq == 199:
			out1_syn_net_np = out1_syn_net.cpu().detach().numpy()
			out2_syn_net_np = out2_syn_net.cpu().detach().numpy()
			sdr = [0.0,0.0]
			for i in range(batch_size):
				out1_syn_net_one = out1_syn_net_np[i,:,:,:]
				out2_syn_net_one = out2_syn_net_np[i,:,:,:]
				spec_input_one = spec_input[i,:,:,:]
				esti=np.r_[out1_syn_net_one*spec_input_one, out2_syn_net_one*spec_input_one] #(2,256,256)
				#print('esti.shape',esti.shape)
				wav_estimated = np.stack([spec2wave(esti[0, :, :]),spec2wave(esti[1, :, :])], axis=0) #(2,nsample)
				#print('wav_estimated.shape',wav_estimated.shape) #(2,nsample)
				wav_gt = np.stack([spec2wave(spec_input1[i, 0, :, :]),spec2wave(spec_input2[i, 0, :, :])], axis=0)
				#print('wav_gt.shape',wav_gt.shape)
				[sdr,sir,sar]=compute_sdr(wav_gt,wav_estimated)
				if not sdr[0] == 100:
					sdr += sdr 
			print('validation:sdr1=%.3f sdr2=%.3f' %(sdr[0]/batch_size,sdr[1]/batch_size))
	#mask1_ratio.tofile('/home/zhc/the_sound/gt1.np')
	#mask2_ratio.tofile('/home/zhc/the_sound/gt2.np')
	#out1_syn_net.cpu().detach().numpy().tofile('/home/zhc/the_sound/1.np')
	#out2_syn_net.cpu().detach().numpy().tofile('/home/zhc/the_sound/2.np')
	if not os.path.exists(model_dir):
		os.mkdir(model_dir)
	torch.save(video_net.state_dict(), os.path.join(model_dir, 'video_net_params.pkl'))
	torch.save(audio_net.state_dict(), os.path.join(model_dir, 'audio_net_params.pkl'))
	torch.save(syn_net.state_dict(), os.path.join(model_dir, 'syn_net_params.pkl'))
	print("model saved to " + str(model_dir) + '\n')

if __name__ == '__main__':
	#spec_dir = 'D:/stddzy/Sound_of_Pixels/audio_mini'
	#image_dir = 'D:/stddzy/Sound_of_Pixels/video_mini'
	#spec_dir = '/home/zhc/the_sound/audio_mini'
	#image_dir = '/home/zhc/the_sound/video_mini'
	spec_dir = '/home/zhc/the_sound/audio_spectrums'
	image_dir = '/home/zhc/the_sound/video_frames'
	model_dir = '/home/zhc/the_sound/model_params_ratio'
	train(spec_dir, image_dir, model_dir)
