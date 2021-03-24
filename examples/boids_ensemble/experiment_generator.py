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
	name = 'experiment';
	runs = 1;
	repeats = 1;
	steps = 10;
	model = '';
	output_location = '';
	file_name = "output.txt"
	generator = None;

	def __init__(self, *args, **kwargs):
		if len(kwargs.items())==0:
			self.name = args[0];
			#self.model = args[1];
			#self.steps = args[2];
			#self.runs = args[3];
		else:
			self.name = kwargs["name"];
			self.output_location = kwargs["output_location"];
			self.filename = kwargs["filename"];
			self.steps = kwargs["steps"];
			self.model = kwargs["model"];
			self.repeats = kwargs["repeats"];
			self.runs = kwargs["runs"];

	def setOutputDirectory(self, loc):
		self.output_location = loc;

	def setOutputFile(self, name):
		self.filename = name;

	def setSimulationSteps(self, steps):
		self.steps = steps;

	def setModel(self, model):
		self.model = model;

	def setRepeats(self, repeats):
		self.repeats = repeats;

	def setRuns(self, runs):
		self.runs = runs;
		print(self.runs);

	def initialStateGenerator(self, generator):
		self.generator = generator;

	def begin(self):
		print("starting experiment");

class InitialStateGenerator(object):
	file = '0.xml';
	global_list = [];
	agent_list = [];

	def __init__(self, *args, **kwargs):
		print("initial states");

	def initialStateFile(self, file):
		self.file = file;
		
	def setGlobalFloat(self, global_name, global_range, generate_random=False, distribution=random.uniform):
		if generate_random or type(global_range)==type(list):
			variable = (global_name, distribution(global_range[0], global_range[1]));
		else:
			variable = (global_name, global_range);
		self.global_list.append(variable);
		print("added float global",global_name);

	def setGlobalInt(self, global_name, global_range, generate_random=False, distribution=random.randint):
		if generate_random or type(global_range)==type(list):
			variable = (global_name, distribution(global_range[0], global_range[1]));
		else:
			variable = (global_name, global_range);
		self.global_list.append(variable);
		print("added float global",global_name);
		
	def setGlobalList(self, global_name, global_range):
		variable = (global_name, global_range);
		self.global_list.append(variable);
		print("added float global",global_name);
		
	def setGlobalString(self, global_name, global_range):
		variable = (global_name, global_range);
		self.global_list.append(variable);
		print("added float global",global_name);

	def setGlobalRandom(self, global_name, global_range, distribution=random.uniform):
		variable = (global_name, distribution(global_range[0], global_range[1]));
		self.global_list.append(variable);
		print("added random global",global_name);

	def addAgentPopulation(self, agent):
		print(agent.name);
		self.agent_list.append(agent);
	
class AgentPopulation(object):
	name = 'agent';
	default_state = 'DEFAULT1';
	pop_min = 1;
	pop_max = 1024;
	pop_list = []
	variable_list = [];

	def __init__(self, *args, **kwargs):
		if len(kwargs.items())==0:
			self.name = args[0];
		else:
			self.name = kwargs["agent"];

	def setDefaultState(self, state):
		self.default_state = state;
		print(state);

	def setPopSize(self, pop_size):
		self.pop_min = pop_size;
		self.pop_max = pop_size;
		print("pop size",pop_size);

	# Population size randomly generated within range
	def setPopSizeMin(self, min_pop):
		self.pop_min = min_pop;
		print(min_pop);

	# Population size randomly generated within range
	def setPopSizeMax(self, max_pop):
		self.pop_max = max_pop;
		print(max_pop);

	def setPopSizeList(self, pop_sizes):
		self.pop_list = pop_sizes;
		print(pop_sizes);

	def setPopSizeRandom(self, range, distribution=random.uniform):
		pop = distribution(range[0],range[1]);
		self.pop_min = pop;
		self.pop_max = pop;
		print(range);

	def setVariableFloat(self, variable_name, variable_range, generate_random=False, distribution=random.uniform):
		if generate_random or type(variable_range)==type(list):
			variable = (variable_name, distribution(variable_range[0], variable_range[1]), True);
		else:
			variable = (variable_name, variable_range, True);
		self.variable_list.append(variable);
		print("added agent variable",variable_name);

	def setVariableInt(self, variable_name, variable_range, generate_random=False, distribution=random.randint):
		if generate_random or type(variable_range)==type(list):
			variable = (variable_name, distribution(variable_range[0], variable_range[1]), True);
		else:
			variable = (variable_name, variable_range, True);
		self.variable_list.append(variable);
		print("added agent variable",variable_name);

	def setVariableList(self, variable_name, variable_range):
		variable = (variable_name, variable_range, True);
		self.variable_list.append(variable);
		print("added agent variable",variable_name);

	def setVariableString(self, variable_name, variable_range):
		variable = (variable_name, variable_range, True);
		self.variable_list.append(variable);
		print("added agent variable",variable_name);

	def setVariableRandomPerAgent(self, variable_name, variable_range, distribution=random.uniform):
		variable = (variable_name, variable_range, False);
		self.variable_list.append(variable);
		print("added agent variable",variable_name);