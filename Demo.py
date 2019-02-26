import os
import cv2
import numpy as np
from dataHelper.Utils import *
import json
import time

from separate_and_locate import Separate_and_Locate

def Test(imagepath,audiopath,OutputJSONPath,OutputAudio,isFromImage=False,VideoPath=''): 

	imagelist=os.listdir(imagepath)
	#print('imagelist',imagelist) #存图片的文件夹名
	result={}
	time_list=[]
	audio_num = 0
	for file in imagelist:
		audio_num += 1
		file_key=file+'.mp4'
		result[file_key]=[]
		imagename=os.path.join(imagepath,file)
		audioname=os.path.join(audiopath,file)+'.wav'
		#print('audioname',audioname)
		#print('imagename',imagename)
		if isFromImage:
			#print('read from image file')
			video,audio=load_data_from_image_file(imagename,audioname)
		else:
			#print('read form video')
			Videoname=os.path.join(VideoPath,file_key)
			video,audio=load_data_from_video(Videoname,audioname,10)

		# print video.shape
		# print audio.shape
		# audio decompostion
		start = time.clock()
		#音源分离与定位
		audio_name = Separate_and_Locate(video, audio, file, OutputAudio)
		#print(audio_name)
		position = [0, 1] #我们并不是没有做定位，基于上面的神经网络模型
		print(position)   #分离出来的第一个音源一定对应图片的左边，因此这里只需要将position置为[0,1]
		
		end = time.clock()
		time_list.append(end-start)
		for i in range(len(audio_name)):
			temp={}
			temp['position']=position[i]
			temp['audio']=audio_name[i]
			result[file_key].append(temp)
	with open(os.path.join(OutputPath,"result.json"),"w") as f:
		json.dump(result,f,indent=4)
	print("test time:",sum(time_list))

if __name__ == '__main__':
	ImageFilePath="testimage"
	VideoPath="testvideo"
	AudioPath="gt_audio"
	OutputPath="result_json/"
	OutputAudio="result_audio/"
	if not os.path.exists(OutputPath):
		os.mkdir(OutputPath)
	if not os.path.exists(OutputAudio):
		os.mkdir(OutputAudio)
	Test(ImageFilePath,AudioPath,OutputPath,OutputAudio,True,VideoPath)