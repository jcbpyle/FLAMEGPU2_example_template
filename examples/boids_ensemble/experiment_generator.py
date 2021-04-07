from pyflamegpu import *
import os
import random
import itertools
import sys
import threading
import queue
import datetime
import numpy as np
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
	file_name = "output.txt"
	generator = None;

	def __init__(self, *args, **kwargs):
		if len(kwargs.items())==0:
			args_items = len(args.items());
			if args_items>3:
				self.runs = args[3];
			if args_items>2:
				self.steps = args[2];
			if args_items>1:
				self.steps = args[2];
			if args_items==1:
				self.name = args[0];
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
	r"""This class allows users to define the construction of a valid inital state for a FLAME-GPU2 model including global variables and agent populations
	TODO: User defined logging, multiple initial state files for experiment, Exceptions of invalid tuple/list input for variables during random generation"""

	#Register default internal values
	file = '0.xml';
	global_list = [];
	agent_list = [];

	def __init__(self, *args, **kwargs):
		print("initial states");

	def initialStateFile(self, file):
		r"""Allows user to specifically set the initial state to an existing file via path
		:type name: string
		:param name: Initial state file name path"""
		self.file = file;
		
	def setGlobalFloat(self, global_name, global_range, generate_random=False, distribution=random.uniform):
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
		if generate_random or type(global_range)==type(tuple):
			variable = (global_name, distribution(global_range[0], global_range[1]));
		else:
			variable = (global_name, global_range);
		self.global_list.append(variable);
		print("added float global",global_name);

	def setGlobalInt(self, global_name, global_range, generate_random=False, distribution=random.randint):
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
		if generate_random or type(global_range)==type(tuple):
			variable = (global_name, distribution(global_range[0], global_range[1]));
		else:
			variable = (global_name, global_range);
		self.global_list.append(variable);
		print("added float global",global_name);
		
	def setGlobalList(self, global_name, global_range):
		r"""Allows user to provide a list of values to assign to a global variable. This will assign the global variable the same list in each simulation instance
		:type global_name: string
		:param global_name: Name of the global variable
		:type global_range: list
		:param global_range: Set global to the current list of values
		"""
		variable = (global_name, global_range);
		self.global_list.append(variable);
		print("added float global",global_name);
		
	def setGlobalString(self, global_name, global_range):
		r"""Allows user to provide a string to assign to a global variable
		:type global_name: string
		:param global_name: Name of the global variable
		:type global_range: string
		:param global_range: Set global to the provided string
		"""
		variable = (global_name, global_range);
		self.global_list.append(variable);
		print("added float global",global_name);

	def setGlobalRandom(self, global_name, global_range, distribution=random.uniform):
		r"""Allows user to provide a range from which to generate a float value randomly to assign to a global variable
		:type global_name: string
		:param global_name: Name of the global variable
		:type global_range: tuple
		:param global_range: Set global randomly each simulation instance via a tuple containing a minimum and maximum value
		"""
		variable = (global_name, distribution(global_range[0], global_range[1]));
		self.global_list.append(variable);
		print("added random global",global_name);

	def addAgentPopulation(self, agent):
		r"""Allows user to add a description of how to generate a specific population of agents to the initial state generator
		:type agent: py:class:`AgentPopulation`
		:param agent: Description of how to generate a valid population of a specific agent"""
		print(agent.name);
		self.agent_list.append(agent);
	
class AgentPopulation(object):
	r"""This class provides an interface to a reproducible parameter search experiment for a FLAME-GPU2 model"""

	#Register default internal values
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
		if generate_random or type(variable_range)==type(tuple):
			variable = (variable_name, distribution(variable_range[0], variable_range[1]), True);
		else:
			variable = (variable_name, variable_range, True);
		self.variable_list.append(variable);
		print("added agent variable",variable_name);

	def setVariableInt(self, variable_name, variable_range, generate_random=False, distribution=random.randint):
		if generate_random or type(variable_range)==type(tuple):
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


def create_individual(container):
		global curr_pop
		new = [0]*(len(self.parameter_limits)+1)
		for i in range(len(self.parameter_limits)):
			if type(parameter_limits[i][0])==type(int()):
				new[i] = int(random.uniform(self.parameter_limits[i][0], self.parameter_limits[i][1]))
			else:
				new[i] = round(random.uniform(self.parameter_limits[i][0], self.parameter_limits[i][1]),6)
		new[-1] = curr_pop
		curr_pop += 1
		new = np.array(new, dtype=np.float64).reshape(1,-1)
		return container(new)

def favour_offspring(parents, offspring, MU):
	choice = (list(zip(parents, [0]*len(parents))) +
				list(zip(offspring, [1]*len(offspring))))
	choice.sort(key=lambda x: ((x[0].fitness.values[0]), x[1]), reverse=True)
	return [x[0] for x in choice[:MU]], [x[0] for x in choice[:MU] if x[1]==1]

def log(logbook, population, gen, evals):
	global statistics
	record = statistics.compile(population) if statistics else {}
	logbook.record(generation=gen,evaluations=evals,**record)
	return

def evaluate_population(population):
	evaluated_population = population
	return evaluated_population

#Define a function for crossover between 2 individuals (many are available in deap if individuals are in bitstring form)
def mate(parent1, parent2):
	global toolbox
	child = toolbox.individual()
	return child

#Define a function for mutating an individual (many are available in deap if individuals are in bitstring form)
def mutate(individual):
	global toolbox

	return individual,

#Define a function for selecting parents (many are available in deap)
def select_parents(individuals,k):
	global toolbox
	#Example selection function, randomly select 2 parents from population
	#parents = [random.choice(individuals) for i in range(k)]
	#return [toolbox.clone(ind) for ind in parents]


class Search(object):
	search_type="GA";
	mu=16;
	lamda=16;
	max_generations=100;
	max_time=100;
	mutation_rate=0.2;
	optimal_fitness=1.0;
	fitness_weights=(1.0,);
	fitness_function=None;
	parameter_limits=None;
	output_file="search_results.csv";
	cwd=os.cwd()+"/";
	logged_stats=["mean", "std", "min", "max"];

	def __init__(*args, **kwargs):
		kwargs_items = kwargs.items();
		if "mu" in kwargs_items:
			mu = kwargs_items["mu"];

	def GA(self):
		global curr_pop, statistics, toolbox
		if not os.path.exists(self.cwd+"ga_temp/"):
			os.mkdir(self.cwd+"ga_temp/")
		working_directory = self.cwd+"ga_temp/"
		if not os.path.exists(working_directory+"optimal_solutions_discovered.csv"):
			open(working_directory+"optimal_solutions_discovered.csv","w").close()
		#Create a fitness function +ve for maximisation, -ve for minimisation
		creator.create("Fitness",base.Fitness,weights=self.fitness_weights)
		creator.create("Individual",list,fitness=creator.Fitness)
		toolbox = base.Toolbox()
		toolbox.register("individual",create_individual,creator.Individual)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		#New statistics should be created for each fitness value to be tracked (and the log method and required globals altered accordingly)
		statistics = tools.Statistics(lambda individual: individual.fitness.values[0])
		for s in self.logged_statistics:
			method = getattr(np,s)
			statistics.register(s,np.method)
		logbook = tools.Logbook()
		logbook.header = ['generation', 'evaluations'] + (statistics.fields if statistics else [])
		toolbox.register("select_parents", select_parents)
		toolbox.register("mutate",mutate)
		toolbox.register("mate",mate)

		current_generation = 0
		#Initialise a population of mu individuals
		population = toolbox.population(n=self.mu)
		start_time = datetime.datetime.now()
		print("Initial population evalauation (Generation 0)")
		#Evaluate initial population
		initial_fitnesses = evaluate_population(population)
		candidates_evaluated = self.mu
		#Record results per GA in file named the same the current seed being used for the random module
		unique_run_seed = random.randrange(sys.maxsize)
		seed_record = working_directory+str(unique_run_seed)+".csv"
		if not os.path.exists(seed_record):
			population_record = open(seed_record,"w")
		else:
			population_record = open(working_directory+str(unique_run_seed)+"(1).csv","w")
		population_record.write("generation,0,mu,"+str(self.mu)+",lambda,"+str(self.lamda)+"\n")
		#Set popualtion fitness to evaluated fitness
		for i in range(len(initial_fitnesses)):
			population[i].fitness.values = initial_fitnesses[i][0]
			population_record.write("\tParameters,")
			for j in population[i][0].tolist():
				population_record.write(str(j)+",")
			population_record.write("Fitness,"+str(population[i].fitness.values)+"\n")
		population_record.close()
		#Record initial population in the logbook
		log(logbook, population, current_generation, self.mu)

		#Begin generational GA process
		end_conditions = False
		optimal_solutions = []
		optimal_count = 0
		while(current_generation<self.max_generations and (not end_conditions)):
			current_generation += 1
			print("\t Generation:",current_generation)
			generational_evaluations = 0
			curr_pop = 0
			offspring = []
			evaluations = []
			#Generate offspring candidates. If crossover is being used, it is done before mutation
			for i in range(self.lamda):
				mate_chance = random.uniform(0,1)
				if mate_chance<mates_rate and (not crossover):
					child = toolbox.individual()
				else:
					parent1, parent2 = [toolbox.clone(x) for x in toolbox.select_parents(population, 2)]
					child = toolbox.mate(parent1, parent2)
				offspring += [child]
			#Mutate new candidates
			for off in offspring:
				off, = toolbox.mutate(off)
			generational_evaluations += len(offspring)
			evaluations = evaluate_population(offspring)
			for i in range(len(evaluations)):
				offspring[i].fitness.values = evaluations[i][0]
			candidates_evaluated += generational_evaluations
			#Select the next generation, favouring the offspring in the event of equal fitness values
			population, new_individuals = favour_offspring(population, offspring, self.mu)
			#Print a report about the current generation
			if generational_evaluations>0:
				log(logbook, population, current_generation, generational_evaluations)
			#Save to file in case of early exit
			#log_fitness = open(working_directory+"current_ga_fitness_log.csv","w")
			#log_fitness.write(str(logbook)+"\n")
			#log_fitness.close()
			if not os.path.exists(seed_record):
				population_record = open(seed_record,"w")
			else:
				population_record = open(working_directory+str(unique_run_seed)+"(1).csv","w")
			check_nonunique = []
			for p in population:
				population_record.write("\t")
				for q in p[0].tolist():
					population_record.write(str(q)+",")
				population_record.write("fitness,"+str(p.fitness.values)+",fitnesses,"+str(p.fitnesses.values)+"\n")
				if p.fitness.values[0]>optimal_fitness:
					for opt in optimal_solutions:
						check_nonunique.append(all(elem in p[0][:-1] for elem in opt[0][:-1]))
					if not any(check_nonunique):
						optimal_solutions.append((p,current_generation))
			population_record.write("SimulationGA,generation,"+str(current_generation)+"\n")
			population_record.close()
			end_time = datetime.datetime.now()
			time_taken = end_time-start_time
			if time_taken>=self.max_time:
				end_conditions=True;
			opti = optimal_solutions[optimal_count:]
			if len(opti)>0:
				opt = open(working_directory+"optimal_solutions_discovered.csv","a")
				for b in opti:
					opt.write("SimulationGAseed,"+str(unique_run_seed)+",Solution_Parameters,"+str(b[0][0].tolist())+",Fitness,"+str(b[0].fitness.values)+",Discovered_Generation,"+str(b[1])+",Discovered_Time,"+str(end_time)+"\n")
				opt.close()
			optimal_count = len(optimal_solutions)

		#Record GA results
		if not os.path.exists(self.cwd+output_file):
			results_file = open(self.cwd+output_file,"w")
		results_file = open(self.cwd+results_file,"a")
		results_file.write(str(logbook)+"\n")
		results_file.close()
		if not os.path.exists(self.cwd+"ga_times.csv"):
			open(loc+"times.csv","w").close()
		time = open(loc+"times.csv","a")
		time.write("ga_seed,"+str(unique_run_seed)+",started_at,"+str(start_time)+",ended_at,"+str(end_time)+",total_time,"+str(time_taken)+"\n")
		time.close()