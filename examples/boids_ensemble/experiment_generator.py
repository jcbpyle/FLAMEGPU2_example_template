from pyflamegpu import *
import os
import random
import itertools
import sys
import threading
import queue
import datetime
import numpy as np


class Experiment(object):
	r"""This class provides an interface to a reproducible parameter search experiment for a FLAME-GPU2 model"""

	#Register default internal variables for an experiment created with no arguments
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

	def setModelLogDirectory(self, loc):
		r"""Allows user to specifically set the expected directory for model logs
		:type loc: string
		:param loc: Path to the model log directory"""
		self.output_location = loc;

	def setModelLogFile(self, name):
		r"""Allows user to specifically set the expected file name for model logs
		:type name: string
		:param name: Expected name of model log file"""
		self.filename = name;

	def setSimulationSteps(self, steps):
		r"""Allows user to specifically set the number of steps to run each model simulation within the experiment for
		:type steps: uint
		:param steps: Simulation steps"""
		self.steps = steps;

	def setModel(self, model):
		r"""Allows user to specifically set the FLAME-GPU2 model to be used in the experiment
		:type model: py:class:`pyflamegpu.ModelDescription`
		:param model: FLAME-GPU2 model"""
		self.model = model;

	def setRepeats(self, repeats):
		r"""Allows user to specifically set the number of times to simulate each set of initial parameter values
		:type steps: uint
		:param steps: Repeat simulations per set of initial parameter values"""
		self.repeats = repeats;

	def setRuns(self, runs):
		r"""Allows user to specifically set the number of steps to run each model simulation within the experiment for
		:type steps: uint
		:param steps: Simulation steps"""
		self.runs = runs;
		print(self.runs);

	def initialStateGenerator(self, generator):
		r"""Allows user to set the generator to be used in initial population creation for a simulation
		:type model: py:class:`InitialStateGenerator`
		:param model: Initial population and global value generator"""
		self.generator = generator;

	def begin(self):
		r"""Begin the experiment with current experiment values"""
		print("Beginning experiment");
		if (self.runs>1):
			simulation = pyflamegpu.CUDAEnsemble(self.model);
			run_plan_vector = pyflamegpu.RunPlanVec(self.model, self.runs);
			run_plan_vector.setSteps(self.steps);
			simulation_seed = random.randint(0,99999);
			run_plan_vector.setRandomSimulationSeed(simulation_seed,1000);
		else:
			simulation = pyflamegpu.CUDASimulation(model);
		simulation.initialise(sys.argv);

		if (self.runs>1):
			simulation.simulate(run_plan_vector);
		else:
			simulation.simulate();

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