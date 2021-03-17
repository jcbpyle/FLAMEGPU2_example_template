import os
import random
import itertools
import sys
import threading
import queue
import datetime
import numpy as np

BASE_DIRECTORY = os.getcwd()+"/";
PROJECT_DIRECTORY = BASE_DIRECTORY;
OS_NAME = os.name;
PROJECT_DIRECTORY = BASE_DIRECTORY+"/";

class Experiment(object):

	def __init__(self, experiment_name):
		self.name = experiment_name;

class InitialStateGenerator(object):

	def __init__(self):
		print("");
		
	def addGlobalFloat(self, global_name, global_range, random_type):
		print("added float global",global_name);

	def addGlobalInt(self, global_name, global_range, random_type):
		print("added float global",global_name);
		
	def addGlobalList(self, global_name, global_range, random_type):
		print("added float global",global_name);
		
	def addGlobalString(self, global_name, global_range, random_type):
		print("added float global",global_name);
		
	def addAgent