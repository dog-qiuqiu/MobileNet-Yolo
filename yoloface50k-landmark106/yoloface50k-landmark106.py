# -*- coding: UTF-8 -*-
from __future__  import division
import math
import caffe
import numpy as np
import cv2
from collections import Counter
import time,os

#nms算法
def nms(dets, thresh=0.35):
	#dets:N*M,N是bbox的个数，M的前4位是对应的（x1,y1,x2,y2），第5位是对应的分数
	#thresh:0.3,0.5....
	x1 = dets[:, 0]
	y1 = dets[:, 1]
	x2 = dets[:, 2]
	y2 = dets[:, 3]
	scores = dets[:, 4]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)#求每个bbox的面积
	order = scores.argsort()[::-1]#对分数进行倒排序
	keep = []#用来保存最后留下来的bbox
	while order.size > 0:
		i = order[0]#无条件保留每次迭代中置信度最高的bbox
		keep.append(i)
		#计算置信度最高的bbox和其他剩下bbox之间的交叉区域
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])
		#计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		#求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
		ovr = inter / (areas[i] + areas[order[1:]] - inter)
		#保留ovr小于thresh的bbox，进入下一次迭代。
		inds = np.where(ovr <= thresh)[0]
		#因为ovr中的索引不包括order[0]所以要向后移动一位
		order = order[inds + 1]
	return keep

#定义sigmod函数
def sigmod(x):
  	return 1.0 / (1.0 + math.exp(-x))

#检测模型前向运算
def Load_YOLO_model(net,test_img):
	input_img = cv2.resize(test_img,(INPUT_SIZE,INPUT_SIZE),interpolation=cv2.INTER_AREA)
	input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
	input_img = input_img.transpose(2,0,1)
	input_img = input_img.reshape((1,3,INPUT_SIZE,INPUT_SIZE))
	out = net.forward_all(data=input_img/256.)
	return out["layer33-conv"].transpose(0, 3, 2, 1)[0]

#处理前向输出feature_map
def feature_map_handle(length, shape, test_img, box_list):
	ih,iw,_ = test_img.shape
	confidence = 0.75
	for i in range(length):
		for j in range(length):
			anchors_boxs_shape = shape[i][j].reshape((3, 6))
			#将每个预测框向量包含信息迭代出来
			for k in range(3):
				anchors_box = anchors_boxs_shape[k]
				#计算实际置信度,阀值处理,anchors_box[7]
				score = sigmod(anchors_box[4])
				if score > confidence:
					#tolist()数组转list
					cls_list = anchors_box[5:6].tolist()
					label = cls_list.index(max(cls_list))
					obj_score = score
					x = ((sigmod(anchors_box[0]) + i)/float(length))*iw
					y = ((sigmod(anchors_box[1]) + j)/float(length))*ih

					w = (((BIAS_W[k]) * math.exp(anchors_box[2]))/INPUT_SIZE)*iw
					h = (((BIAS_H[k]) * math.exp(anchors_box[3]))/INPUT_SIZE)*ih
					x1 = int(x - w * 0.5)
					x2 = int(x + w * 0.5)
					y1 = int(y - h * 0.2)
					y2 = int(y + h * 0.5)
					box_list.append([x1,y1,x2,y2,round(obj_score,4),label])


#3个feature_map的预选框的合并及NMS处理
def dect_box_handle(out_shape, test_img):
	box_list = []
	output_box = []
	length =  len(out_shape)
	feature_map_handle(length, out_shape, test_img, box_list)
	#print box_list
	if box_list:
		retain_box_index = nms(np.array(box_list))
		for i in retain_box_index:
			output_box.append(box_list[i])
	return output_box

def forward_landmark(landmark_net,face_roi,bbox):
	ih,iw, _ = face_roi.shape
	sw = float(iw)/float(112)
	sh = float(ih)/float(112)
	
	res = cv2.resize(face_roi, (112, 112), 0.0, 0.0, interpolation=cv2.INTER_CUBIC)
	resize_mat = np.float32(res)
	#(x-127.5)/127.5
	new_img = (resize_mat - 127.5) / (127.5)
	input_shape = new_img.transpose(2,0,1)
	out = landmark_net.forward_all(data=input_shape)
	points = landmark_net.blobs['bn6_3'].data[0].flatten()
	for i in range(int(len(points)/2)):
		points[i*2] = ((points[i*2]*112)*sw)+bbox[0]
		points[(i*2)+1] = ((points[(i*2)+1]*112)*sh)+bbox[1]
	return points
		
def draw_point(img, points):
	for i in range(int(len(points)/2)):
		cv2.circle(img,(int(points[i*2]),int(points[(i*2)+1])), 1, (255,255,255), 1)

def test_img():	
	#加载检测模型
	net = caffe.Net('caffemodel/yoloface-50k.prototxt', 'caffemodel/yoloface-50k.caffemodel', caffe.TEST)
	landmark_net = caffe.Net("caffemodel/landmark106.prototxt", "caffemodel/landmark106.caffemodel", caffe.TEST)
	#################################
	test_img = cv2.imread("test.jpeg")
	out_shape = Load_YOLO_model(net,test_img)
	output_box = dect_box_handle(out_shape, test_img)
	for i in output_box:
		face_roi = test_img[i[1]:i[3],i[0]:i[2]]
		points = forward_landmark(landmark_net,face_roi,i)
		draw_point(test_img, points)
		cv2.rectangle(test_img, (i[0], i[1]), (i[2], i[3]), (255, 255, 0), 2)
		cv2.circle(test_img, (int(i[0]+0.5*(i[2]-i[0])), int(i[1]+0.5*(i[3]-i[1]))), 2, (0,0,255), 3)			
		#cv2.putText(test_img, "Score:"+str(i[4]), (i[0], i[1]-5), 0, 0.7, (255, 0, 255), 2)	
		#cv2.putText(test_img, "Label:"+"Face", (i[0], i[1]-20), 0, 0.7, (255, 255, 0), 2)
	cv2.imwrite("yoloface-50k-landmark106.jpg",test_img)
	

def test_cam():
	#加载检测模型
	net = caffe.Net('caffemodel/yoloface-50k.prototxt', 'caffemodel/yoloface-50k.caffemodel', caffe.TEST)
	landmark_net = caffe.Net("caffemodel/landmark106.prototxt", "caffemodel/landmark106.caffemodel", caffe.TEST)
	#################################
        cap = cv2.VideoCapture(0)     
        while True:
		ret, test_img = cap.read()
		out_shape = Load_YOLO_model(net,test_img)
		output_box = dect_box_handle(out_shape, test_img)
		for i in output_box:
			face_roi = test_img[i[1]:i[3],i[0]:i[2]]
			points = forward_landmark(landmark_net,face_roi,i)
			draw_point(test_img, points)
			cv2.rectangle(test_img, (i[0], i[1]), (i[2], i[3]), (255, 255, 0), 1)
			cv2.circle(test_img, (int(i[0]+0.5*(i[2]-i[0])), int(i[1]+0.5*(i[3]-i[1]))), 2, (0,0,255), 3)			
			#cv2.putText(test_img, "Score:"+str(i[4]), (i[0], i[1]-5), 0, 0.7, (255, 0, 255), 1)	
		cv2.imshow("test", test_img)
		cv2.waitKey(1)	
	cap.release()
	cv2.destroyAllWindows()	

if __name__ == "__main__":
	INPUT_SIZE = 56
	#模型训练时设置的anchor_box比例
	#7, 10,  12, 17,  22, 27
	BIAS_W = [7, 12, 22]
	BIAS_H = [12, 19, 29]
	#摄像头检测
	#test_cam()
	#图像检测
	test_img()
