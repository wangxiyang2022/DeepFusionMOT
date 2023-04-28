# -*-coding:utf-8-*
# author: wangxy
import cv2

'''
	This code is converting images to video.
'''
# Write your folder path here，example：/home/youname/data/img/
# Note that the last folder should have a /
img_root = '../results/pointrcnn_Car_val/image/'
video_save_path = 'tracking_video.avi'
fps = 24
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter(video_save_path,fourcc,fps,(1242,375))

for i in range(297):  # Here 297 is the number of frames in the dataset. You need to make the appropriate changes
	number = '%06d'%i
	frame = cv2.imread(img_root+number+'.png')
	videoWriter.write(frame)
videoWriter.release()
