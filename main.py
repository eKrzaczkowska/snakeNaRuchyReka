#---------------------------------------BIBLIOTEKI------------------------------------------
import cv2
import copy
import numpy as np
import ctypes as C
import math
import os
import time
from random import shuffle 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy
import math
import multiprocessing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
import pygame
from pygame.locals import *
import random, sys
model = load_model('model.h5')

#---------------------------------def gra-----------------------------------------------
global width, rows, s, jedzenie
action = ''
wynik =''
width = 500
rows = 20

class kwadrat(object):
    rows = 20
    w = 500
    def __init__(self,start,dirnx=1,dirny=0,color=(255,0,0)):
        self.pos = start
        self.dirnx = 1
        self.dirny = 0
        self.color = color
 
       
    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny
        self.pos = (self.pos[0] + self.dirnx, self.pos[1] + self.dirny)
 
    def draw(self, surface, eyes=False):
        dis = self.w // self.rows
        i = self.pos[0]
        j = self.pos[1]
 
        pygame.draw.rect(surface, self.color, (i*dis+1,j*dis+1, dis-2, dis-2))
        if eyes:
            centre = dis//2
            radius = 3
            circleMiddle = (i*dis+centre-radius,j*dis+8)
            circleMiddle2 = (i*dis + dis -radius*2, j*dis+8)
            pygame.draw.circle(surface, (0,0,0), circleMiddle, radius)
            pygame.draw.circle(surface, (0,0,0), circleMiddle2, radius)
       
 
class waz(object):
	body = []
	turns = {}
	def __init__(self, color, pos):
		self.color = color
		self.head = kwadrat(pos)
		self.body.append(self.head)
		self.dirnx = 0
		self.dirny = 1
 

 
	def reset(self, pos):
		self.head = kwadrat(pos)
		self.body = []
		self.body.append(self.head)
		self.turns = {}
		self.dirnx = 0
		self.dirny = 1


	def move(self):

		
		if wynik=='turnleft':
			self.dirnx = -1
			self.dirny = 0
			self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
		
		elif wynik=='turnright':
			self.dirnx = 1
			self.dirny = 0
			self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
		
		elif wynik=='onefinger':
			self.dirnx = 0
			self.dirny = -1
			self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
		
		elif wynik=='fist':
			self.dirnx = 0
			self.dirny = 1
			self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

		for i, c in enumerate(self.body): 
			p = c.pos[:]  
			if p in self.turns:  
				turn = self.turns[p]  
				c.move(turn[0],turn[1])  
				if i == len(self.body)-1:  
					self.turns.pop(p)
			else:  
				c.move(c.dirnx,c.dirny)  
 
	def addCube(self):
		tail = self.body[-1]
		dx, dy = tail.dirnx, tail.dirny
 
		if dx == 1 and dy == 0:
			self.body.append(kwadrat((tail.pos[0]-1,tail.pos[1])))
		elif dx == -1 and dy == 0:
			self.body.append(kwadrat((tail.pos[0]+1,tail.pos[1])))
		elif dx == 0 and dy == 1:
			self.body.append(kwadrat((tail.pos[0],tail.pos[1]-1)))
		elif dx == 0 and dy == -1:
			self.body.append(kwadrat((tail.pos[0],tail.pos[1]+1)))
 
		self.body[-1].dirnx = dx
		self.body[-1].dirny = dy
       
 
	def draw(self, surface):
		for i, c in enumerate(self.body):
			if i ==0:
				c.draw(surface, True)
			else:
				c.draw(surface)
 
 
def siatka(w, rows, okno):
    sizeBtwn = w // rows
 
    x = 0
    y = 0
    for l in range(rows):
        x = x + sizeBtwn
        y = y + sizeBtwn
 
        pygame.draw.line(okno, (255,255,255), (x,0),(x,w))
        pygame.draw.line(okno, (255,255,255), (0,y),(w,y))
       
 
def redrawWindow(okno):
    global rows, width, s, jedzenie
    okno.fill((0,0,0))
    s.draw(okno)
    jedzenie.draw(okno)
    siatka(width,rows, okno)
    pygame.display.update()
 
 
def losujeJedzenie(rows, obiekt):
 
    positions = obiekt.body
 
    while True:
        x = random.randrange(rows)
        y = random.randrange(rows)
        if len(list(filter(lambda z:z.pos == (x,y), positions))) > 0:
            continue
        else:
            break
       
    return (x,y)
 

#----------------------------------zmienne-----------------------------------------------
#----------General Settings-----------------------------------------
prediction = ''
score = 0
img_counter = 500
filename = 0
LABELS = {5:'turnright', 4:'turnleft', 3:'onefinger', 2:'ok', 1:'hand', 0:'fist'}
rows=20

#------------wyodrebnianie reki z obrazu-----------------------------
x_begin = 0.5  # start point/total width
y_end = 0.6  # start point/total width

#-----------przetwaranie obrazu------------------
isBgCaptured = 0  # bool, whether the background captured
blurValue = 31  # GaussianBlur parameter
learningRate = 0
threshold = 60  # binary threshold


#--------------------------------program ----------------------------------------------------
#--------------------------okno gry--------------------
running = True
win = pygame.display.set_mode((width, width))
s = waz((255,0,0), (10,10))
jedzenie = kwadrat(losujeJedzenie(rows, s), color=(0,255,0))
clock=pygame.time.Clock()
#--------------------------kamera--------------
camera = cv2.VideoCapture(0)
camera.set(10, 200)
if isBgCaptured==0:
	subtractor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=30, detectShadows=False)
	isBgCaptured = 1

while running:
	pygame.time.delay(60)
	clock.tick(10)
	s.move()
	if s.body[0].pos == jedzenie.pos:
            s.addCube()
            jedzenie = kwadrat(losujeJedzenie(rows, s), color=(0,255,0))
 
	for x in range(len(s.body)):
            if s.body[x].pos in list(map(lambda z:z.pos,s.body[x+1:])):
                print('Score: ', len(s.body))
                s.reset((10,10))
                break

	if action == 'turnleft' and s.head.pos[0] < 0: 
		s.reset((10,10))
	elif action == 'turnright' and s.head.pos[0] >= rows: 
		s.reset((10,10))
	elif action == 'onefinger' and s.head.pos[1] < 0: 
		s.reset((10,10))
	elif action == 'fist' and s.head.pos[1] >= rows: 
		s.reset((10,10))
	elif action == 'hand' and (s.head.pos[1] >= rows)|(s.head.pos[0] < 0)|(s.head.pos[0] >= rows)|(s.head.pos[1] < 0): 
		s.reset((10,10))
           
	redrawWindow(win)
#---------------------------pobieranie obrazu z kamery----------------------------------------
	_, frame = camera.read()
	frame = cv2.bilateralFilter(frame, 5, 50, 100) 
	frame = cv2.flip(frame, 1) 
	cv2.rectangle(frame, (int(x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(y_end * frame.shape[0])), (255, 0, 0), 2)
#-----------------------------wyodrebnienie dloni----------------------------------------
	if isBgCaptured == 1:
		#-----------------------wyodrebnainie dloni------------------------------
		img = subtractor.apply(frame, learningRate=0)
		dlon = cv2.bitwise_and(frame, frame, mask=img)
		dlon = dlon[0:int(y_end * frame.shape[0]),
		int(x_begin * frame.shape[1]):frame.shape[1]]
		#-------------------------skala szarosci----------------------------------
		szarosc = cv2.cvtColor(dlon, cv2.COLOR_BGR2GRAY)
		#-------------------------rozmywanie--------------------------------------
		rozmazane= cv2.GaussianBlur(szarosc, (blurValue, blurValue), 0)
		#--------------------------czarno bialy-----------------------------------
		ret, koniec = cv2.threshold(rozmazane, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		#---------------------------PRZEWIDYWANIE Z MODELU------------------------
		przewidywany=koniec
		przewidywany = cv2.resize(przewidywany, (28,28))
		przewidywany=np.array(przewidywany.astype('float32'))
		przewidywany=przewidywany/255.0
		img_array = np.array(przewidywany)
		img_array = np.expand_dims(img_array, axis=0)
		img_array = np.expand_dims(img_array, axis=4)
		prediction = model.predict(img_array)
		maximum=np.argmax(prediction) 
		action = LABELS[maximum]
		if action != wynik:
			wynik = action
		print(action)
		cv2.imshow('dlon', koniec)
		#-----------------------------koniec dzialania-------------------------
		if action == 'ok':
			#running=False
			print("wykryto ok koniec gry")
		#----------------------------------gra----------------------------------
		for event in pygame.event.get():
			if event.type==pygame.QUIT:
				running=False
				print("koniec gry")
		#--------------------------skroty klawiszowe---------------------------		
		k = cv2.waitKey(10)
		#--------------------------zapisywani obrazu---------------------------
		if k == ord('s'):
			directory = '/home/ewa-laptop/Desktop/sztuczna inteligencja/valid/fist'
			path = directory+"//"+str(0)+"_"+str(filename)+".png"
			filename += 1
			img = cv2.resize(thresh, (28,28))
			mask=np.array(img.astype('float32'))
			mask=mask/255.0
			cv2.imwrite(path, mask)
		#---------------------------koniec dzialania programu-----------------
		elif k == ord('q'):
			break

cv2.destroyAllWindows()
