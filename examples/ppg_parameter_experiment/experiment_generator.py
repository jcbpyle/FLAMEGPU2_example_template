from pyflamegpu import *
import os
import random
import itertools
import sys
import threading
import queue
import datetime
import numpy as np
import copy
from deap import base
from deap import creator
from deap import tools


class Experiment(object):
	r"""This class provides an interface to a reproducible parameter search experiment for a FLAME-GPU2 model"""

	#Register default internal variables for an experiment created with no arguments
	name = 'experiment';
	runs = 1;
	repeats = 1;
	steps = 10;
	model = '';
	output_location = '';
	filename = 'output.txt'
	generator = None;
	log = None;
	sim_log = None;
	simulation = None;
	verbose = False;

	def __init__(self, *args, **kwargs):
		if len(kwargs)==0:
			args_items = len(args);
			if args_items>4:
				print("Please use keywords such as 'model=...' for more detailed initialisation.");
			if args_items>3:
				self.repeats = args[3];
			if args_items>2:
				self.steps = args[2];
			if args_items>1:
				self.runs = args[1];
			if args_items>=1:
				self.name = args[0];
		else:
			self.name = kwargs['name'] if ('name' in kwargs) else self.name;
			self.output_location = kwargs['output_location'] if ('output_location' in kwargs) else self.output_location;
			self.filename = kwargs['filename'] if ('filename' in kwargs) else self.filename;
			self.steps = kwargs['steps'] if ('steps' in kwargs) else self.steps;
			self.model = kwargs['model'] if ('model' in kwargs) else self.model;
			self.repeats = kwargs['repeats'] if ('repeats' in kwargs) else self.repeats;
			self.runs = kwargs['runs'] if ('runs' in kwargs) else self.runs;
			self.generator = kwargs['generator'] if ('generator' in kwargs) else self.generator;
			self.verbose = kwargs['verbose'] if ('verbose' in kwargs) else self.verbose;

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

	def setLog(self, log):
		r""" """
		self.log = log;

	def setRepeats(self, repeats):
		r"""Allows user to specifically set the number of times to run simulations from each set of initial parameter values 
		:type repeats: uint
		:param repeats: Repeat simulations per set of initial parameter values"""
		self.repeats = repeats;

	def setRuns(self, runs):
		r"""Allows user to specifically set the number of times to run simulations from each set of initial parameter values 
		:type runs: uint
		:param runs: Repeat simulations per set of initial parameter values"""
		self.runs = runs;

	def initialStateGenerator(self, generator):
		r"""Allows user to set the generator to be used in initial population creation for a simulation
		:type model: py:class:`InitialStateGenerator`
		:param model: Initial population and global value generator"""
		self.generator = generator;

	def begin(self):
		r"""Begin the experiment with current experiment values"""
		if (self.verbose):
			print("Beginning experiment");
		if (self.runs>1):
			if (self.verbose):
				print("Preparing experiment ensemble run plan vector");
			self.simulation = pyflamegpu.CUDAEnsemble(self.model);
			run_plan_vector = pyflamegpu.RunPlanVec(self.model, self.runs);
			run_plan_vector.setSteps(self.steps);
			simulation_seed = random.randint(0,sys.maxsize);
			seed_step = random.randint(1,sys.maxsize/self.runs);
			run_plan_vector.setRandomSimulationSeed(simulation_seed,seed_step);
		else:
			if (self.verbose):
				print("Performing single simulation experiment, with steps", self.steps);
			self.simulation = pyflamegpu.CUDASimulation(self.model);
			self.simulation.SimulationConfig().steps = self.steps
		if not self.log==None:
			step_log = pyflamegpu.StepLoggingConfig(self.log);
			step_log.setFrequency(1);
			self.simulation.setStepLog(step_log)
			self.simulation.setExitLog(self.log)
		
		if (self.runs>1):
			if (self.verbose):
				print("Beginning experiment ensemble");
			self.simulation.simulate(run_plan_vector);
			if not self.log==None:
				if (self.verbose):
					print("ensemble logging")
				self.sim_log = self.simulation.getLogs();
				for log in self.sim_log:
					steps = log.getStepLog();
					if (self.verbose):
						for step in steps:
							print()
							print("prey",step.getAgent("Prey").getCount());
							print("predator",step.getAgent("Predator").getCount());
							print(type(self.sim_log))
		else:
			self.simulation.simulate();
			if not self.log==None:
				self.sim_log = self.simulation.getRunLog();
				if (self.verbose):
					print(type(self.sim_log))			
					print(type(self.sim_log.getExitLog()))
					print(self.sim_log.getExitLog().getEnvironmentPropertyArrayFloat("fitnesses"))
					#self.sim_log = simulation.getRunLog().getExitLog().getEnvironmentPropertyArrayFloat("fitnesses")
		if (self.verbose):
			print("Completed experiment:", self.name);

class InitialStateGenerator(object):
	r"""This class allows users to define the construction of a valid inital state for a FLAME-GPU2 model including global variables and agent populations
	TODO: User defined logging, multiple initial state files for experiment, Exceptions of invalid tuple/list input for variables during random generation"""

	#Register default internal values
	file = None;
	global_list = [];
	agent_list = [];

	def __init__(self, *args, **kwargs):
		if len(kwargs)==0:
			if len(args)>=1:
				self.file = args[0] if type(args[0])==type('') else None;
		else:
			self.file = kwargs['file'] if ('file' in kwargs and type(kwargs['file'])==type('')) else self.file;

	def initialStateFile(self, file):
		r"""Allows user to specifically set the initial state to an existing file via path
		:type name: string
		:param name: Initial state file name path"""
		self.file = file if type(file)==type('') else None;

	def __setVariable(self, global_name, global_range, distribution=random.uniform):
		variable_names = [var[0] for var in self.global_list];
		if global_name in variable_names:
			variable_update = True;
			variable_index = variable_names.index(global_name);
		else:
			variable_update = False;
		if type(global_range)==type(tuple()):
			variable = (global_name, distribution(global_range[0], global_range[1]));
		else:
			variable = (global_name, global_range);
		if variable_update:
			self.global_list[variable_index] = variable;
		else:
			self.global_list.append(variable);
		
	def setGlobalFloat(self, global_name, global_range, distribution=random.uniform):
		r"""Allows user to describe the initialisation of a global float parameter as a constant value, list of values, or tuple of minimum and maximum values from which to randomly generate a value.
		*Overload 1:*
		:type global_name: string
		:param global_name: Name of the global variable
		:type global_range: float
		:param global_range: Set global to a constant float value to be used over the experiment
		:type generate_random: boolean
		:param generate_random: User may specify that the value should be generated randomly. Defaults to False, intended to be used when user provides a tuple of minimum and maximum range values
		:type distribution: py:class:`random.uniform`
		:param distribution: User may specify a function via which to generate the value. Defaults to python's random.uniform

		*Overload 2:*
		:type global_name: string
		:param global_name: Name of the global variable
		:type global_range: list
		:param global_range: Set global to the current list of values
		:type generate_random: boolean
		:param generate_random: User may specify that the value should be generated randomly. Defaults to False, intended to be used when user provides a tuple of minimum and maximum range values
		:type distribution: py:class:`random.uniform`
		:param distribution: User may specify a function via which to generate the value. Defaults to python's random.uniform

		*Overload 3:*
		:type global_name: string
		:param global_name: Name of the global variable
		:type global_range: tuple
		:param global_range: Set global via a tuple containing a minimum and maximum value
		:type generate_random: boolean
		:param generate_random: User may specify that the value should be generated randomly. Defaults to False, intended to be used when user provides a tuple of minimum and maximum range values
		:type distribution: py:class:`random.uniform`
		:param distribution: User may specify a function via which to generate the value. Defaults to python's random.uniform
		"""
		self.__setVariable(global_name, global_range, distribution);

	def setGlobalInt(self, global_name, global_range, distribution=random.randint):
		r"""Allows user to describe the initialisation of a global integer parameter as a constant value, list of values, or tuple of minimum and maximum values from which to randomly generate a value.
		*Overload 1:*
		:type global_name: string
		:param global_name: Name of the global variable
		:type global_range: uint
		:param global_range: Set global to a constant int value to be used over the experiment
		:type generate_random: boolean
		:param generate_random: User may specify that the value should be generated randomly. Defaults to False, intended to be used when user provides a tuple of minimum and maximum range values
		:type distribution: py:class:`random.randint`
		:param distribution: User may specify a function via which to generate the value. Defaults to python's random.randint

		*Overload 2:*
		:type global_name: string
		:param global_name: Name of the global variable
		:type global_range: list
		:param global_range: Set global to the current list of values
		:type generate_random: boolean
		:param generate_random: User may specify that the value should be generated randomly. Defaults to False, intended to be used when user provides a tuple of minimum and maximum range values
		:type distribution: py:class:`random.randint`
		:param distribution: User may specify a function via which to generate the value. Defaults to python's random.randint

		*Overload 3:*
		:type global_name: string
		:param global_name: Name of the global variable
		:type global_range: tuple
		:param global_range: Set global via a tuple containing a minimum and maximum value
		:type generate_random: boolean
		:param generate_random: User may specify that the value should be generated randomly. Defaults to False, intended to be used when user provides a tuple of minimum and maximum range values
		:type distribution: py:class:`random.randint`
		:param distribution: User may specify a function via which to generate the value. Defaults to python's random.randint
		"""
		self.__setVariable(global_name, global_range, distribution);
		
	def setGlobalList(self, global_name, global_range):
		r"""Allows user to provide a list of values to assign to a global variable. This will assign the global variable the same list in each simulation instance
		:type global_name: string
		:param global_name: Name of the global variable
		:type global_range: list
		:param global_range: Set global to the current list of values
		"""
		self.__setVariable(global_name, global_range);
		
	def setGlobalString(self, global_name, global_range):
		r"""Allows user to provide a string to assign to a global variable
		:type global_name: string
		:param global_name: Name of the global variable
		:type global_range: string
		:param global_range: Set global to the provided string
		"""
		self.__setVariable(global_name, global_range);

	def setGlobalRandom(self, global_name, global_range, distribution=random.uniform):
		r"""Allows user to provide a range from which to generate a float value randomly to assign to a global variable
		:type global_name: string
		:param global_name: Name of the global variable
		:type global_range: tuple
		:param global_range: Set global randomly each simulation instance via a tuple containing a minimum and maximum value
		"""
		self.__setVariable(global_name, global_range, distribution);

	def addAgentPopulation(self, agent):
		r"""Allows user to add a description of how to generate a specific population of agents to the initial state generator
		:type agent: py:class:`AgentPopulation`
		:param agent: Description of how to generate a valid population of a specific agent"""
		self.agent_list.append(agent);
	
class AgentPopulation(object):
	r"""This class provides an interface to a reproducible parameter search experiment for a FLAME-GPU2 model"""

	#Register default internal values
	name = 'agent';
	agent_state = 'DEFAULT1';
	pop_min = 1;
	pop_max = 1024;
	pop_list = []
	variable_list = [];

	def __init__(self, *args, **kwargs):
		if len(kwargs.items())==0:
			args_items = len(args);
			if args_items>=1:
				self.name = args[0];
		else:
			self.name = kwargs["agent"] if ("agent" in kwargs) else self.name;

	def setDefaultState(self, state):
		r"""Allows user to set the state for initialised agents within the population
		:type state: string
		:param state: Name of the state for agents to be initialised in
		"""
		self.agent_state = state;

	def setPopSize(self, pop_size):
		r"""Allows user to set the desired size of the population via minimum and maximum values
		*Overload 1:*
		:type pop_size: uint
		:param pop_size: Set the minimum and maximum population sizes to a single value

		*Overload 2:*
		:type pop_size: tuple
		:param pop_size: Set the minimum and maximum population sizes via tuple values provided by the user
		"""
		if (type(pop_size)==type(tuple())):
			self.pop_min = pop_size[0];
			self.pop_max = pop_size[1];
		else:
			self.pop_min = pop_size;
			self.pop_max = pop_size;

	def setPopSizeMin(self, min_pop):
		r"""Allows user to set the minimum number of agents to be generated for the current population
		:type min_pop: uint
		:param min_pop: Set the minimum population size to the desired value
		"""
		self.pop_min = min_pop;

	def setPopSizeMax(self, max_pop):
		r"""Allows user to set the maximum number of agents to be generated for the current population
		:type max_pop: uint
		:param max_pop: Set the maximum population size to the desired value
		"""
		self.pop_max = max_pop;

	def setPopSizeList(self, pop_sizes):
		r"""Allows user to provide a list of population size values, each of which is used to generate a population of agents
		:type pop_sizes: list
		:param pop_sizes: Set a list of population sizes to be generated
		"""
		self.pop_list = pop_sizes;

	def setPopSizeRandom(self, pop_range, distribution=random.randint):
		r"""Allows user to set the a range for the population size, from which a single value will be randomly generated
		:type pop_range: tuple
		:param pop_range: Provide the minimum and maximum allowed population sizes
		"""
		pop = distribution(pop_range[0],pop_range[1]);
		self.pop_min = pop;
		self.pop_max = pop;

	def __setVariable(self, variable_name, variable_range, distribution=random.uniform):
		variable_names = [var[0] for var in self.variable_list];
		if variable_name in variable_names:
			variable_update = True;
			variable_index = variable_names.index(variable_name);
		else:
			variable_update = False;
		if type(variable_range)==type(tuple()):
			variable = (variable_name, distribution(variable_range[0], variable_range[1]), True);
		else:
			variable = (variable_name, variable_range, True);
		if variable_update:
			self.variable_list[variable_index] = variable;
		else:
			self.variable_list.append(variable);

	def setVariable(self, variable_name, variable_range):
		if type(variable_range)==type(float()):
			self.__setVariable(variable_name, variable_range);
		else:
			if type(variable_range)==type(int()):
				self.__setVariable(variable_name, variable_range);
			else:
				print("Invalid use of setVariable(), please provide a variable name, and value as either int or float type\n");

	def setVariableFloat(self, variable_name, variable_range, distribution=random.uniform):
		r"""Allows user to describe the initialisation of a float variable for *all* agents within the population as a constant value or tuple of minimum and maximum values from which to randomly generate a value.
		*Overload 1:*
		:type variable_name: string
		:param variable_name: Name of the agent variable
		:type variable_range: float
		:param variable_range: Set variable value to a constant float value for all agents in the population
		:type generate_random: boolean
		:param generate_random: User may specify that the value should be generated randomly. Defaults to False, intended to be used when user provides a tuple of minimum and maximum range values
		:type distribution: py:class:`random.uniform`
		:param distribution: User may specify a function via which to generate the value. Defaults to python's random.uniform

		*Overload 2:*
		:type variable_name: string
		:param variable_name: Name of the global variable
		:type variable_range: tuple
		:param variable_range: Randomly generate variable value via a tuple containing a minimum and maximum value, set this value for all agents in the population
		:type generate_random: boolean
		:param generate_random: User may specify that the value should be generated randomly. Defaults to False, intended to be used when user provides a tuple of minimum and maximum range values
		:type distribution: py:class:`random.uniform`
		:param distribution: User may specify a function via which to generate the value. Defaults to python's random.uniform
		"""
		self.__setVariable(variable_name, variable_range, distribution);

	def setVariableInt(self, variable_name, variable_range, distribution=random.randint):
		r"""Allows user to describe the initialisation of a integer variable for *all* agents within the population as a constant value or tuple of minimum and maximum values from which to randomly generate a value.
		*Overload 1:*
		:type variable_name: string
		:param variable_name: Name of the agent variable
		:type variable_range: int
		:param variable_range: Set variable value to a constant float value for all agents in the population
		:type generate_random: boolean
		:param generate_random: User may specify that the value should be generated randomly. Defaults to False, intended to be used when user provides a tuple of minimum and maximum range values
		:type distribution: py:class:`random.randint`
		:param distribution: User may specify a function via which to generate the value. Defaults to python's random.randint

		*Overload 2:*
		:type variable_name: string
		:param variable_name: Name of the global variable
		:type variable_range: tuple
		:param variable_range: Randomly generate variable value via a tuple containing a minimum and maximum value
		:type generate_random: boolean
		:param generate_random: User may specify that the value should be generated randomly. Defaults to False, intended to be used when user provides a tuple of minimum and maximum range values
		:type distribution: py:class:`random.randint`
		:param distribution: User may specify a function via which to generate the value. Defaults to python's random.randint
		"""
		self.__setVariable(variable_name, variable_range, distribution);

	def setVariableList(self, variable_name, variable_range):
		r"""Allows user to provide a list of values to assign to an agent variable. This will set the list of values for *all* agents in the population
		:type variable_name: string
		:param variable_name: Name of the agent variable
		:type variable_range: list
		:param variable_range: Set agent variable value to the provided list of values
		"""
		variable = (variable_name, variable_range, True);
		variable_names = [var[0] for var in self.variable_list];
		if variable_name in variable_names:
			variable_index = variable_names.index(variable_name);
			self.variable_list[variable_index] =variable;
		else:
			self.variable_list.append(variable);

	def setVariableString(self, variable_name, variable_range):
		r"""Allows user to provide a string value to assign to an agent variable. This will set the value for *all* agents in the population
		:type variable_name: string
		:param variable_name: Name of the agent variable
		:type variable_range: string
		:param variable_range: Set agent variable value to the provided string
		"""
		variable = (variable_name, variable_range, True);
		variable_names = [var[0] for var in self.variable_list];
		if variable_name in variable_names:
			variable_index = variable_names.index(variable_name);
			self.variable_list[variable_index] =variable;
		else:
			self.variable_list.append(variable);

	def setVariableRandomPerAgent(self, variable_name, variable_range, distribution='random.uniform'):
		r"""Allows user to describe the initialisation of a float variable for *each* agent within the population via a tuple of minimum and maximum values from which to randomly generate a value
		:type variable_name: string
		:param variable_name: Name of the agent variable
		:type variable_range: tuple
		:param variable_range: Set variable value for each agent via minimum and maximum values provided
		:type distribution: py:class:`random.uniform`
		:param distribution: User may specify a function via which to generate the value. Defaults to python's random.uniform
		"""
		variable = (variable_name, variable_range, distribution, False);
		variable_names = [var[0] for var in self.variable_list];
		if variable_name in variable_names:
			variable_index = variable_names.index(variable_name);
			self.variable_list[variable_index] =variable;
		else:
			self.variable_list.append(variable);

class Search(object):
	r"""This class provides an interface to a genetic algorithm(GA) based search experiment intended to be used for a FLAME-GPU2 model"""

	#Register default internal values
	search_type="GA";
	#GA population values
	mu=2;
	lamda=1;
	#GA end conditions
	max_generations=5;
	max_time=100;
	optimal_fitness=1.0;
	#GA generational process values
	ga_individual_id=int(0);
	mutation_rate=0.2;
	random_initialisation_chance=0.05;
	#GA fitness calculation
	fitness_weights=(1.0,);
	fitness_function=None;
	#GA chromosome initialisation and limiter
	parameter_limits=[(-1.0,1.0)];
	#GA logging
	output_file="search_results.csv";
	cwd=os.getcwd()+"/";
	logged_stats=["mean", "std", "min", "max"];
	evaluator_experiment = None;
	verbose = False;

	def __init__(*args, **kwargs):
		kwargs_items = kwargs.items();
		if "mu" in kwargs_items:
			mu = kwargs_items["mu"];

	def setPopEvaluationExperiment(self, evaluator):
		r"""Allows user to specifically set a FLAME-GPU2 ensemble experiment to be used in evaluating the fitness of a population of GA individuals
		:type evaluator: py:class:`experiment_generator.Experiment`
		:param evaluator: FLAME-GPU2 ensemble experiment"""
		self.evaluator_experiment = evaluator;

	def __create_individual(self, container):
		r"""Creates a new GA individual (chromosome) with values randomly generated based on the provided limits for each parameter
		:type container: 'deap.creator.Individual'
		:param container: A function assigning the created individual as an individual in a DEAP genetic algorithm population
		"""
		new = [0]*(len(self.parameter_limits)+1)
		for i in range(len(self.parameter_limits)):
			if type(self.parameter_limits[i][0])==type(int()):
				new[i] = int(random.randint(self.parameter_limits[i][0], self.parameter_limits[i][1]))
			else:
				new[i] = round(random.uniform(self.parameter_limits[i][0], self.parameter_limits[i][1]),6)
		new[-1] = int(self.ga_individual_id)
		self.ga_individual_id += 1
		new = np.array(new, dtype=np.float64).reshape(1,-1)
		return container(new)

	def __favour_offspring(self, parents, offspring, MU):
		r"""Evaluates GA population sorted by fitness, favouring the more recently created individuals if fitnesses are equivalent
		:type parents: list
		:param parents: The current GA population
		:type offspring: list
		:param offspring: The newly created individuals this generation
		"""
		choice = (list(zip(parents, [0]*len(parents))) +
					list(zip(offspring, [1]*len(offspring))))
		choice.sort(key=lambda x: ((x[0].fitness.values[0]), x[1]), reverse=True)
		return [x[0] for x in choice[:self.mu]], [x[0] for x in choice[:self.mu] if x[1]==1]

	def __log(self, logbook, population, gen, evals):
		r"""Logs GA values including generation, average population fitness, population maximum and minimum fitness values, and population fitness standard deviation
		:type logbook: 'deap.base.Toolbox.Logbook'
		:param logbook: The logbook recording information over the course of a GA run
		:type population: list
		:param population: The current GA popualtion
		:type gen: int
		:param gen: The current GA generation
		:type evals: int
		:param evals: The number of fitness evaluations performed so far
		"""
		global statistics
		record = statistics.compile(population) if statistics else {}
		logbook.record(generation=gen,evaluations=evals,**record)
		return

	def __evaluate_population(self, population):
		r"""Placeholder fitness evaluation function, should be replaced by user created function to determine fitness
		:type population: list
		:param population: The current GA population
		"""
		n = len(population)
		evaluation = [0.0]*n
		if not self.evaluator_experiment==None:
			for i in range(n):
				self.evaluator_experiment.generator.setGlobalInt("PREY_POPULATION_TO_GENERATE", population[i][0][0])
				self.evaluator_experiment.generator.setGlobalInt("PREDATOR_POPULATION_TO_GENERATE", population[i][0][1])
				self.evaluator_experiment.generator.setGlobalInt("GRASS_POPULATION_TO_GENERATE", population[i][0][2])
				log = self.evaluator_experiment.begin()
				final_log = log.getExitLog()
				fitness_log = final_log.getEnvironmentPropertyArrayFloat("fitnesses")
				evaluation[i] = fitness_log[0]+(fitness_log[1]*100)-(float(fitness_log[2])/1000)
		return evaluation

	def __mate(self, parent1, parent2):
		r"""A function for crossover between 2 GA individuals (many are available in deap if individuals are in bitstring form)
		:type parent1: list
		:param parent1: The first individual selected for crossover
		:type parent2: list
		:param parent2: The second individual selected for crossover
		"""
		global toolbox
		child = toolbox.individual()
		return child

	def __mutate(self, individual):
		r"""A function for mutating an individual (many are available in deap if individuals are in bitstring form)
		:type individual: list
		:param individual: The first individual selected for mutation
		"""
		global toolbox

		return individual,

	def __select_parents(self, population, function=None):
		r"""Define a function for selecting parents (many are available in deap). Defaults to random selection if no choice is provided
		:type function: list
		:param function: User provided deap function for parent selection
		:type population: list
		:param population: The current GA population
		"""
		global toolbox
		if function==None:
			#Example selection function, randomly select 2 parents from population
			parents = [random.choice(population) for i in range(2)]
		return [toolbox.clone(ind) for ind in parents]

	def GA(self):
		r"""Genetic algorithm setup and runner based on current internal values. User should set all values and functions before the call to GA"""
		global statistics, toolbox
		if not os.path.exists(self.cwd+"ga_temp/"):
			os.mkdir(self.cwd+"ga_temp/")
		working_directory = self.cwd+"ga_temp/"
		if not os.path.exists(working_directory+"optimal_solutions_discovered.csv"):
			open(working_directory+"optimal_solutions_discovered.csv","w").close()
		#Create a fitness function +ve for maximisation, -ve for minimisation
		creator.create("Fitness",base.Fitness,weights=self.fitness_weights)
		creator.create("Individual",list,fitness=creator.Fitness)
		toolbox = base.Toolbox()
		toolbox.register("individual",self.__create_individual,creator.Individual)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		#New statistics should be created for each fitness value to be tracked (and the log method and required globals altered accordingly)
		statistics = tools.Statistics(lambda individual: individual.fitness.values[0])
		for s in self.logged_stats:
			statmethod = getattr(np,s)
			statistics.register(s,statmethod)
		logbook = tools.Logbook()
		logbook.header = ['generation', 'evaluations'] + (statistics.fields if statistics else [])
		toolbox.register("select_parents", self.__select_parents)
		toolbox.register("mutate",self.__mutate)
		toolbox.register("mate",self.__mate)

		current_generation = 0
		#Initialise a population of mu individuals
		population = toolbox.population(n=self.mu)
		start_time = datetime.datetime.now()
		if (self.verbose):
			print("Initial population evalauation (Generation 0)")
		#Evaluate initial population
		initial_fitnesses = self.__evaluate_population(population);
		if (self.verbose):
			print("initial_fitnesses",initial_fitnesses)
		candidates_evaluated = self.mu
		#Record results per GA in file named the same the current seed being used for the random module
		unique_run_seed = random.randrange(sys.maxsize)
		seed_record = working_directory+str(unique_run_seed)+".csv"
		if os.path.exists(seed_record):
			seed_record = working_directory+str(unique_run_seed)+"(1).csv"
			population_record = open(seed_record,"w")
		else:
			population_record = open(seed_record,"a")
		population_record.write("SimulationGA,generation,0,mu,"+str(self.mu)+",lambda,"+str(self.lamda)+"\n")
		#Set population fitness to evaluated fitness
		for i in range(len(initial_fitnesses)):
			population[i].fitness.values = (initial_fitnesses[i],)
			population_record.write("\tChromosome_ID,")
			population_record.write(str(int(population[i][0][-1]))+",")
			population_record.write("Parameters,")
			for j in range(len(population[i][0].tolist())-1):
				population_record.write(str(int(population[i][0][j]))+",")
			population_record.write("Fitness,"+str(population[i].fitness.values[0])+"\n")
		population_record.close()
		#Record initial population in the logbook
		self.__log(logbook, population, current_generation, self.mu)
		if (self.verbose):
			print("begin GA");
		#Begin generational GA process
		end_conditions = False
		optimal_solutions = []
		optimal_count = 0
		while(not end_conditions):
			current_generation += 1
			if (self.verbose):
				print("GA Generation:",current_generation)
			generational_evaluations = 0
			offspring = []
			evaluations = []
			population_record = open(seed_record, "a")
			population_record.write("SimulationGA,generation,"+str(current_generation)+"\n")
			population_record.close()
			#Generate offspring candidates. If crossover is being used, it is done before mutation
			for i in range(self.lamda):
				mate_chance = random.uniform(0,1)
				if mate_chance<self.random_initialisation_chance:
					child = toolbox.individual()
				else:
					parent1, parent2 = [toolbox.clone(x) for x in toolbox.select_parents(population)]
					child = toolbox.mate(parent1, parent2)
				offspring += [child]
			#Mutate new candidates
			for off in offspring:
				off, = toolbox.mutate(off)
			generational_evaluations += len(offspring)
			evaluations = self.__evaluate_population(offspring);
			for i in range(len(evaluations)):
				offspring[i].fitness.values = (evaluations[i],)
			candidates_evaluated += generational_evaluations
			#Select the next generation, favouring the offspring in the event of equal fitness values
			population, new_individuals = self.__favour_offspring(population, offspring, self.mu)
			#Print a report about the current generation
			if generational_evaluations>0:
				self.__log(logbook, population, current_generation, generational_evaluations)
			#Save to file in case of early exit
			#log_fitness = open(working_directory+"current_ga_fitness_log.csv","w")
			#log_fitness.write(str(logbook)+"\n")
			#log_fitness.close()
			population_record = open(seed_record, "a")
			check_nonunique = []
			for p in population:
				population_record.write("\tChromosome_ID,")
				population_record.write(str(int(p[0].tolist()[-1]))+",")
				population_record.write("Parameters,")
				for j in range(len(p[0].tolist())-1):
					population_record.write(str(int(p[0][j]))+",")
				population_record.write("Fitness,"+str(p.fitness.values[0])+"\n")
				if p.fitness.values[0]>self.optimal_fitness:
					for opt in optimal_solutions:
						check_nonunique.append(all(elem in p[0][:-1] for elem in opt[0][:-1]))
					if not any(check_nonunique):
						optimal_solutions.append((p,current_generation))
			population_record.close()
			end_time = datetime.datetime.now()
			time_taken = (end_time-start_time).total_seconds();
			minutes_taken = (int)(time_taken/60.0);
			if time_taken>=self.max_time or current_generation>=self.max_generations:
				end_conditions=True;
			opti = optimal_solutions[optimal_count:]
			if len(opti)>0:
				opt = open(working_directory+"optimal_solutions_discovered.csv","a")
				for b in opti:
					opt.write("SimulationGAseed,"+str(unique_run_seed)+",Solution_Parameters,"+str(b[0][0].tolist())+",Fitness,"+str(b[0].fitness.values)+",Discovered_Generation,"+str(b[1])+",Discovered_Time,"+str(end_time)+"\n")
				opt.close()
			optimal_count = len(optimal_solutions)

		#Record GA results
		if not os.path.exists(self.cwd+self.output_file):
			results_file = open(self.cwd+self.output_file,"w")
		results_file = open(self.cwd+self.output_file,"a")
		results_file.write(str(logbook)+"\n")
		results_file.close()
		if not os.path.exists(self.cwd+"search_times.csv"):
			open(self.cwd+"search_times.csv","w").close()
		time = open(self.cwd+"search_times.csv","a")
		time.write("ga_seed,"+str(unique_run_seed)+",started_at,"+str(start_time)+",ended_at,"+str(end_time)+",total_time,"+str(time_taken)+"\n")
		time.close()