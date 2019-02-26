import numpy as np
import librosa
import torch
import os
from librosa import amplitude_to_db
from math import floor
from models import modifyresnet18, UNet, Synthesizer
from util.validation import spec2wave
from image2instru import Instru_from_image
import soundfile as sf
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ori_SAMPLE_RATE = 44100
SAMPLE_RATE = 11000
wave_length = 66302
WINDOW_SIZE = 1022
HOP_LENGTH = 256
frequencies = np.linspace(SAMPLE_RATE/2/512,SAMPLE_RATE/2,512)
log_freq = np.log10(frequencies)
sample_freq = np.linspace(log_freq[0],log_freq[-1],256)
sample_index = np.array([np.abs(log_freq-x).argmin() for x in sample_freq])
model_dir = 'model_params_ratio/'

def Separate_and_Locate(video, audio, file_name, OutputAudio):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	video_net = modifyresnet18().to(device) #载入网络
	audio_net = UNet().to(device)
	syn_net = Synthesizer().to(device)
	video_net.load_state_dict(torch.load(os.path.join(model_dir, 'video_net_params.pkl')))
	audio_net.load_state_dict(torch.load(os.path.join(model_dir, 'audio_net_params.pkl')))
	syn_net.load_state_dict(torch.load(os.path.join(model_dir, 'syn_net_params.pkl')))
	print('params loaded')
	print('video.shape',video.shape) #图片每秒2.4帧
	#print('audio.shape',audio.shape)
	audio = np.reshape(audio,(audio.shape[1]))
	audio = librosa.resample(audio,ori_SAMPLE_RATE,SAMPLE_RATE)
	#print('audio.shape',audio.shape)
	wave_num = floor(audio.shape[0]/wave_length)
	print('wave_num',wave_num)
	wave_seq1 = np.zeros(audio.shape[0], dtype='float32')
	wave_seq2 = np.zeros(audio.shape[0], dtype='float32')
	for i in range(wave_num):
		#print('i=',i)
		data = audio[i*wave_length:(i+1)*wave_length]
		stft_data = librosa.stft(data,n_fft = WINDOW_SIZE,hop_length = HOP_LENGTH,center = False)
		stft_data = stft_data[sample_index,:] #(256,256)
		stft_data_abs = np.absolute(stft_data)
		#试试输入不取db
		spec_input = torch.from_numpy(np.reshape(stft_data_abs,(1,1,256,256))).float().to(device)
		#spec_input = torch.from_numpy(np.reshape(amplitude_to_db(stft_data_abs),(1,1,256,256))).float().to(device)
		#print('spec_input.shape',spec_input.shape)
		image3 = video[i*floor(2.4*wave_length/SAMPLE_RATE):i*floor(2.4*wave_length/SAMPLE_RATE)+3,:,:,:] #(3,224,224,3)
		#print('image3.shape',image3.shape)
		image3_1 = np.zeros((3,224,224,3), dtype='float32')
		image3_2 = np.zeros((3,224,224,3), dtype='float32')
		for j in range(3):
			#print('j=',j)
			image3_1[j,:,:,:] = cv2.resize(image3[j,:,0:112,:],(224,224))
			image3_2[j,:,:,:] = cv2.resize(image3[j,:,112:224,:],(224,224))
		#cv2.namedWindow("Image") #打印图片
		#cv2.imshow("Image", image3_1[1])
		#cv2.waitKey(0)
		#cv2.imwrite('D:/test.jpg',image3_1)
		#print('image3_1.shape',image3_1)
		#print('image3_1.shape',image3_1.shape)
		image3_1 = np.transpose(image3_1,(0,3,2,1)) #(3,224,224,3)->(3,3,224,224)
		image3_1 = np.transpose(image3_1,(0,1,3,2))
		image3_2 = np.transpose(image3_2,(0,3,2,1))
		image3_2 = np.transpose(image3_2,(0,1,3,2))
		#image3_1 = image3[:,:,:,0:112]
		#image3_2 = image3[:,:,:,112:224]
		#print(image3.shape)
		image_input1 = torch.from_numpy(image3_1).float().to(device) #(3,3,224,224)
		image_input2 = torch.from_numpy(image3_2).float().to(device)
		#print('image_input.shape',image_input1.shape)
		out_audio_net = audio_net(spec_input)
		out1_video_net = video_net(image_input1) #送进video网络 (1,16,1,1)
		#print('out_video_net',out1_video_net.shape)
		out2_video_net = video_net(image_input2)
		input1_syn_net = out1_video_net * out_audio_net #送进syn网络之前 (1,16,256,256)
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
		out2_syn_net = torch.transpose(out2_syn_net,1,2) #(1,1,256,256)
		mask1 = out1_syn_net[0,0,:,:].cpu().detach().numpy()
		mask2 = out2_syn_net[0,0,:,:].cpu().detach().numpy()
		#for j in range(mask1.shape[0]):
		#	for k in range(mask1.shape[1]):
		#		if mask1[j,k] >= mask1.mean():
		#			mask1[j,k] = 1
		#		else:
		#			mask1[j,k] = 0
		#		if mask2[j,k] >= mask2.mean():
		#			mask2[j,k] = 1
		#		else:
		#			mask2[j,k] = 0
		#mask1 = np.round(mask1)
		#mask2 = np.round(mask2)
		#print(mask1)
		#print(mask2)
		spec_pre1 = np.multiply(mask1, stft_data)
		spec_pre2 = np.multiply(mask2, stft_data)
		wave_seq1[i*wave_length:(i+1)*wave_length] = spec2wave(spec_pre1)
		wave_seq2[i*wave_length:(i+1)*wave_length] = spec2wave(spec_pre2)
	output_file_name1 = str(file_name) + '_seg1.wav' #wav文件名
	output_file_name2 = str(file_name) + '_seg2.wav'
	audio_name1 = os.path.join(OutputAudio,output_file_name1)
	audio_name2 = os.path.join(OutputAudio,output_file_name2)
	#print(wave_seq1.shape)
	wave_seq1 = librosa.resample(wave_seq1,SAMPLE_RATE,ori_SAMPLE_RATE) #升采样
	wave_seq2 = librosa.resample(wave_seq2,SAMPLE_RATE,ori_SAMPLE_RATE)
	#print('wave_seq.shape',wave_seq1.shape)
	#print(wave_seq1.shape)
	sf.write(audio_name1, wave_seq1, ori_SAMPLE_RATE) #存音频
	sf.write(audio_name2, wave_seq2, ori_SAMPLE_RATE)
	audio_name = [audio_name1, audio_name2]
	print(audio_name)
	return audio_name