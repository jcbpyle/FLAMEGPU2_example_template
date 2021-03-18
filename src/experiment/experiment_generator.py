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

	def __init__(self, *args, **kwargs):
		self.name = experiment_name;
		self.output_location = output_location;
		self.filename = name;
		self.steps = steps;
		self.model = model;
		self.repeats = repeats;

	def setOutputDirectory(self, loc):
		self.output_location = loc;

	def setOutputFile(self, name):
		self.filename = name;

	def setSimulationSteps(self, steps):
		self.steps = steps;

	def model(self, model):
		self.model = model;

	def setRepeats(self, repeats):
		self.repeats = repeats;

class InitialStateGenerator(object):

	def __init__(self, *args, **kwargs):
		print("");
		
	def setGlobalFloat(self, global_name, global_range, random_type):
		print("added float global",global_name);

	def setGlobalInt(self, global_name, global_range, random_type):
		print("added float global",global_name);
		
	def setGlobalList(self, global_name, global_range, random_type):
		print("added float global",global_name);
		
	def setGlobalString(self, global_name, global_range, random_type):
		print("added float global",global_name);

	def setGlobalRandom(self, global_name, global_range, random_type):
		print("added random global",global_name);
	
class AgentPopulation(object):

	def __init__(self, *args, **kwargs):
		self.name = agent_name;

	def setPopulationSize(self, pop_size):
		print("pop size",pop_size);

	def setPopulationSizeList(self, pop_sizes):
		print(pop_sizes);

	def setPopulationSize(self, min_pop, max_pop):
		print(min_pop,max_pop);

	def setVariableFloat(self, variable_name, variable_range, random_type):
		print("added agent variable",variable_name);

	def setVariableInt(self, variable_name, variable_range, random_type):
		print("added agent variable",variable_name);

	def setVariableList(self, variable_name, variable_range, random_type):
		print("added agent variable",variable_name);

	def setVariableRandomPerAgent(self, variable_name, variable_range, random_type):
		print("added agent variable",variable_name);