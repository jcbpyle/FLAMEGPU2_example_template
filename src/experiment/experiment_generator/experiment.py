
# Copyright 2019 The University of Sheffield
# Author: James Pyle
# Contact: jcbpyle1@sheffield.ac.uk
# Template experiment script file for FLAME GPU agent-based model
#
# University of Sheffield retain all intellectual property and 
# proprietary rights in and to this software and related documentation. 
# Any use, reproduction, disclosure, or distribution of this software 
# and related documentation without an express license agreement from
# University of Sheffield is strictly prohibited.
#
# For terms of licence agreement please attached licence or view licence 
# on www.flamegpu.com website.
#


import os
import random
import itertools
import sys
import threading
import queue
import datetime
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

BASE_DIRECTORY = os.getcwd()+"/"
PROJECT_DIRECTORY = BASE_DIRECTORY
GPUS_AVAILABLE = cuda.Device(0).count()
OS_NAME = os.name

PROJECT_DIRECTORY = BASE_DIRECTORY+"/"
#InitialStates

#Generate initial states based on defined ranges/lists/values for all global and agent population variables for experiment predpreygrass.
def generate_initial_states_predpreygrass(location_name=''):
	global_data = []
	agent_data = []
	vary_per_agent = []
	
	global_data = {"EXPERIMENT1":[0],"EXPERIMENT2":[0],"EXPERIMENT3":[0],"EXPERIMENT4":[0],"EXPERIMENT5":[0],"EXPERIMENT6":[0],"EXPERIMENT7":[0],"SMSE_EXPERIMENT":[1]}
	prey = {"initial_population":[50], "type":[3.0],"steer_x":[0.0],"steer_y":[0.0],"age":[0],"parent1_id":[-1],"parent2_id":[-1],"mate_id":[-1],"escaping_flag":[0],"partner_flag":[0],"parent_flag":[0],"speed":[1.0],"agent_sense_distance":[0.1],"cohesion_distance":[0.2],"avoid_distance":[0.035],"food_energy_gain":[35],"kill_distance":[0.02],"reproduction_chance":[0.05]}
	prey_vary_per_agent = {"x":[-1.0,1.0,"uniform",float],"y":[-1.0,1.0,"uniform",float],"fx":[-1.0,1.0,"uniform",float],"fy":[-1.0,1.0,"uniform",float],"energy":[0,50,"uniform",int],"energy_threshold":[0,100,"uniform",int],}
	predator = {"initial_population":[25], "type":[1.0],"age":[0],"parent1_id":[-1],"parent2_id":[-1],"mate_id":[-1],"chasing_prey":[0],"partner_flag":[0],"parent_flag":[0],"speed":[1.0],"chase_speed":[2.0],"food_energy_gain":[5],"agent_sense_distance":[0.1],"avoid_distance":[0.035],"reproduction_chance":[0.025]}
	predator_vary_per_agent = {"x":[-1.0,1.0,"uniform",float],"y":[-1.0,1.0,"uniform",float],"fx":[-1.0,1.0,"uniform",float],"fy":[-1.0,1.0,"uniform",float],"steer_x":[-1.0,1.0,"uniform",float],"steer_y":[-1.0,1.0,"uniform",float],"energy":[0,40,"uniform",int],"energy_threshold":[0,100,"uniform",int],}
	grass = {"initial_population":[250], "type":[2.0],"dead_cycles":[0],"available":[1],"regrow_cycles":[431],"grass_eat_distance":[0.02]}
	grass_vary_per_agent = {"x":[-1.0,1.0,"uniform",float],"y":[-1.0,1.0,"uniform",float],}
	
	agent_data = {"prey":prey,"predator":predator,"grass":grass}
	
	vary_per_agent = {"prey":prey_vary_per_agent,"predator":predator_vary_per_agent,"grass":grass_vary_per_agent}

	
	prefix_components = []
	prefix = ''
	
	prefix = location_name+prefix
	if len(global_data)>0:
		global_names = [x for x in global_data]
		unnamed_global_combinations = list(itertools.product(*[y for x,y in global_data.items()]))
		global_combinations = list(zip([global_names for x in range(len(unnamed_global_combinations))],unnamed_global_combinations))
	if len(agent_data)>0:
		agent_names = [x for x in agent_data]
		unnamed_agent_combinations = list(itertools.product(*[z for x,y in agent_data.items() for w,z in y.items()]))
		loc = 0
		agent_combinations = [[] for x in range(len(unnamed_agent_combinations))]
		for an in agent_names:
			num_vars = loc+len(agent_data[an])
			var_names = [x for x in agent_data[an]]
			sublists = [x[loc:num_vars] for x in unnamed_agent_combinations]
			named_combinations = list(zip([var_names for x in range(len(sublists))],sublists))
			for i in range(len(named_combinations)):
				temp_list = [an]
				temp_list += [[named_combinations[i][0][x],[named_combinations[i][1][x]]] for x in range(len(named_combinations[i][0]))]
				agent_combinations[i] += [temp_list]
			loc = num_vars
	if len(global_combinations)>0 and len(agent_combinations)>0:
		for g in global_combinations:
			for a in agent_combinations:
				current_agent_data = [agent+[[x[0],x[1]] for x in vary_per_agent[agent[0]].items()] for agent in a]
				initial_state("experiment/",str(prefix),"0.xml",g,current_agent_data)
	elif len(global_combinations)>0:
		for g in global_combinations:
			current_agent_data = [agent+[[x[0],x[1]] for x in vary_per_agent[agent[0]].items()] for agent in agent_data]
			initial_state("experiment/",str(prefix),"0.xml",g,current_agent_data)
	elif len(agent_combinations)>0:
		for a in agent_combinations:
			current_agent_data = [agent+[[x[0],x[1]] for x in vary_per_agent[agent[0]].items()] for agent in a]
			initial_state("experiment/",str(prefix),"0.xml",global_data,current_agent_data)
	else:
		print("No global or agent variations specified for experimentation\n")
	return global_data,agent_data,location_name

#Initial state file creation.
def initial_state(save_location,folder_prefix,file_name,global_information,agent_information):
	#if not os.path.exists(PROJECT_DIRECTORY+""):
		#os.mkdir(PROJECT_DIRECTORY+"")
	#SAVE_DIRECTORY = PROJECT_DIRECTORY+""+"/"
	save_split = save_location.split("/")
	temp = ''
	for i in save_split:
		temp += i+"/"
		if not os.path.exists(PROJECT_DIRECTORY+temp):
			os.mkdir(PROJECT_DIRECTORY+temp)
	fp_split = folder_prefix.split("/")
	temp = ''
	for i in fp_split:
		temp += i+"/"
		if not os.path.exists(PROJECT_DIRECTORY+save_location+temp):
			os.mkdir(PROJECT_DIRECTORY+save_location+temp)
	SAVE_DIRECTORY = PROJECT_DIRECTORY+"/"+save_location+"/"+folder_prefix+"/"
	print("Creating initial state in",SAVE_DIRECTORY,"/",file_name,"\n")
	initial_state_file = open(SAVE_DIRECTORY+str(file_name),"w")
	initial_state_file.write("<states>\n<itno>0</itno>\n<environment>\n")
	if len(global_information)>0:
		for g in range(len(global_information[0])):
			initial_state_file.write("<"+str(global_information[0][g])+">"+str(global_information[1][g])+"</"+str(global_information[0][g])+">\n")
	initial_state_file.write("</environment>\n")
	if len(agent_information)>0:
		for i in range(len(agent_information)):
			try:
				ind = [x[0] for x in agent_information[i]].index("initial_population")
			except:
				ind = 0
			num_agents = int(agent_information[i][ind][1][0])
			agent_id = 1
			agent_name = agent_information[i][0]
			for j in range(num_agents):
				initial_state_file.write("<xagent>\n")
				initial_state_file.write("<name>"+str(agent_name)+"</name>\n")
				initial_state_file.write("<id>"+str(agent_id)+"</id>\n")
				for k in agent_information[i]:
					if not (k[0]=="initial_population" or k==agent_name):
						if len(k[1])>1:
							if len(k[1])==4:
								random_method = getattr(random, k[1][2])
								initial_state_file.write("<"+str(k[0])+">"+str(k[1][3](random_method(k[1][0],k[1][1])))+"</"+str(k[0])+">\n")
							elif len(k[1])==3:
								initial_state_file.write("<"+str(k[0])+">"+str(k[1][2](random.uniform(k[1][0],k[1][1])))+"</"+str(k[0])+">\n")
							else:
								initial_state_file.write("<"+str(k[0])+">"+str(random.uniform(k[1][0],k[1][1]))+"</"+str(k[0])+">\n")
						elif type(k[1][0])==type(int()):
							initial_state_file.write("<"+str(k[0])+">"+str(int(k[1][0]))+"</"+str(k[0])+">\n")
						elif type(k[1][0])==type(float()):
							initial_state_file.write("<"+str(k[0])+">"+str(float(k[1][0]))+"</"+str(k[0])+">\n")
						
				initial_state_file.write("</xagent>\n")
				agent_id += 1
	initial_state_file.write("</states>")
	return

#ExperimentSet

############## setup_experiment ############

def setup():
	
	experiment_seed = random.randrange(sys.maxsize)
	random.seed(experiment_seed)
	experiment_start_time = datetime.datetime.now()
	
	if not os.path.exists(PROJECT_DIRECTORY+"/experiment/"):
		os.mkdir(PROJECT_DIRECTORY+"/experiment/")
	experiment_info_file = open(PROJECT_DIRECTORY+"/experiment/experiment_information.csv","w")
	experiment_info_file.write("Experiment,setup\nSeed,"+str(experiment_seed)+"\nstart_time,"+str(experiment_start_time)+"\n")
	experiment_info_file.close()
	open(PROJECT_DIRECTORY+"experiment/simulation_results.csv","w").close()
	
	#Run for desired number of repeats
	REPEATS = 10
	base_output_directory = PROJECT_DIRECTORY+"experiment/"
	for i in range(REPEATS):
		location_name = "setup_"+str(i)+"/"
		generate_initial_states_predpreygrass(location_name)
		#generation_time = datetime.datetime.now()
		#ef = open(PROJECT_DIRECTORY+"experiment/experiment_information.csv","a")
		#ef.write("repeat,"+str(i)+"\ninitial_states_generated,"+str(generation_time)+"\n")
		#ef.close()
		#Model executable
		executable = ""
		simulation_command = ""
		os_walk = list(os.walk(base_output_directory+location_name))
		if len(os_walk[0][1])>1:
			initial_states = [x[0] for x in os_walk][1:]
		else:
			initial_states = [x[0] for x in os_walk]
		for j in initial_states:
			current_initial_state = j+"/0.xml"
			if OS_NAME=='nt':
				executable = PROJECT_DIRECTORY+"//PreyPredator.exe"
				simulation_command = executable+" "+current_initial_state+" 1000"
			else:
				executable = "./"+PROJECT_DIRECTORY+"//PreyPredator"
				simulation_command = executable+" "+current_initial_state+" 1000"
			print(simulation_command)
			#Run simulation
			os.system(simulation_command)

			#Parse results
			results_file = open(j+"/log.csv","r")
			results = results_file.readlines()
			results_file.close()
			sim_results_file = open(PROJECT_DIRECTORY+"experiment/simulation_results.csv","a")
			for res in results:
				sim_results_file.write(res)
			sim_results_file.write("\n")
			sim_results_file.close()
			print(results)
	experiment_completion_time = datetime.datetime.now()
	time_taken = experiment_completion_time-experiment_start_time
	ef = open(PROJECT_DIRECTORY+"experiment/experiment_information.csv","a")
	ef.write("completion_time,"+str(experiment_completion_time)+"\ntime_taken,"+str(time_taken)+"\n")
	
	return 

def main():
	
	#Initial state creation function
	#initial_state(save_directory, initial_state_file_name, initial_state_global_data_list, initial_state_agent_data_list)

	#Generation functions (will automatically call initial state generation function)
	
	#generate_initial_states_predpreygrass()
	
	#Experiment Set user defined functions
	setup()
	
	return

if __name__ == "__main__":
	main()


######################## TEMPLATE SEARCH AND SURROGATE MODELLING CODE ##########################################################

##Template (1+1)GA search
#from deap import base
#from deap import creator
#from deap import tools
#import numpy as np
#import datetime
#import queue
#import threading
#
##Alterable parameters, recommend using larger mu (e.g. 100) to reduce chance of being stuck in local optima and population domination by variations of strong candidate, similar with lambda (e.g. 25)
#mu = 1
#lam = 1
#max_generations = 100
##Maximum run time in minutes
#max_time = 100
#crossover = True
#mutation_rate = 0.2
#mates_rate = 0.5
##Threshold at which a candidate solution is considered optimal
#optimal_fitness = 0.95
##Provide a list with min and max for each parameter 
##parameter_limits = [[parameter1_min,parameter1_max],[parameter2_min,parameter2_max]]
#output_file = "ga_results.csv"
#cwd = os.cwwd()+"/"
#logged_statistics = ["mean", "std", "min", "max"]
#
#def genetic_algorithm(mu,lam,max_generations,max_time,loc,output_file):
#	global curr_pop, statistics, toolbox
#	if not os.path.exists(cwd+"ga_temp/"):
#		os.mkdir(cwd+"ga_temp/")
#	working_directory = cwd+"ga_temp/"
#	if not os.path.exists(working_directory+"optimal_solutions_discovered.csv"):
#		open(working_directory+"optimal_solutions_discovered.csv","w").close()
#	#Create a fitness function +ve for maximisation, -ve for minimisation
#	creator.create("Fitness",base.Fitness,weights=(1.0,))#ALTER THIS FITNESS WEIGHTING (possible to have multiple weightings for multiple values e.g. minimise a and c but maximise b with weightings (-1.0,0.75,-0.5,))
#	creator.create("Individual",list,fitness=creator.Fitness)
#	toolbox = base.Toolbox()
#	toolbox.register("individual",create_individual,creator.Individual)
#	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#	#New statistics should be created for each fitness value to be tracked (and the log method and required globals altered accordingly)
#	statistics = tools.Statistics(lambda individual: individual.fitness.values[0])
#	for s in logged_statistics:
#		method = getattr(np,s)
#		statistics.register(s,np.method)
#	logbook = tools.Logbook()
#	logbook.header = ['generation', 'evaluations'] + (statistics.fields if statistics else [])
#	toolbox.register("select_parents", select_parents)
#	toolbox.register("mutate",mutate)
#	toolbox.register("mate",mate)
#
#	current_generation = 0
#	#Initialise a population of mu individuals
#	population = toolbox.population(n=mu)
#	start_time = datetime.datetime.now()
#	print("Initial population evalauation (Generation 0)")
#	#Evaluate initial population
#	initial_fitnesses = evaluate_population(population)
#	candidates_evaluated = mu
#	#Record results per GA in file named the same the current seed being used for the random module
#	unique_run_seed = random.randrange(sys.maxsize)
#	seed_record = working_directory+str(unique_run_seed)+".csv"
#	if not os.path.exists(seed_record):
#		population_record = open(seed_record,"w")
#	else:
#		population_record = open(working_directory+str(unique_run_seed)+"(1).csv","w")
#	population_record.write("generation,0,mu,"+str(mu)+",lambda,"+str(lam)+"\n")
#	#Set popualtion fitness to evaluated fitness
#	for i in range(len(initial_fitnesses)):
#		population[i].fitness.values = initial_fitnesses[i][0]
#		population_record.write("\tParameters,")
#		for j in population[i][0].tolist():
#			population_record.write(str(j)+",")
#		population_record.write("Fitness,"+str(population[i].fitness.values)+"\n")
#	population_record.close()
#	#Record initial population in the logbook
#	log(logbook, population, current_generation, mu)
#
#	#Begin generational GA process
#	end_conditions = False
#	optimal_solutions = []
#	optimal_count = 0
#	while(current_generation<max_generations and (not end_conditions)):
#		current_generation += 1
#		print("\t Generation:",current_generation)
#		generational_evaluations = 0
#		curr_pop = 0
#		offspring = []
#		evaluations = []
#		#Generate offspring candidates. If crossover is being used, it is done before mutation
#		for i in range(lam):
#			mate_chance = random.uniform(0,1)
#			if mate_chance<mates_rate and (not crossover):
#				child = toolbox.individual()
#			else:
#				parent1, parent2 = [toolbox.clone(x) for x in toolbox.select_parents(population, 2)]
#				child = toolbox.mate(parent1, parent2)
#			offspring += [child]
#		#Mutate new candidates
#		for off in offspring:
#			off, = toolbox.mutate(off)
#		generational_evaluations += len(offspring)
#		evaluations = evaluate_population(offspring)
#		for i in range(len(evaluations)):
#			offspring[i].fitness.values = evaluations[i][0]
#		candidates_evaluated += generational_evaluations
#		#Select the next generation, favouring the offspring in the event of equal fitness values
#		population, new_individuals = favour_offspring(population, offspring, mu)
#		#Print a report about the current generation
#		if generational_evaluations>0:
#			log(logbook, population, current_generation, generational_evaluations)
#		#Save to file in case of early exit
#		#log_fitness = open(working_directory+"current_ga_fitness_log.csv","w")
#		#log_fitness.write(str(logbook)+"\n")
#		#log_fitness.close()
#		if not os.path.exists(seed_record):
#			population_record = open(seed_record,"w")
#		else:
#			population_record = open(working_directory+str(unique_run_seed)+"(1).csv","w")
#		check_nonunique = []
#		for p in population:
#			population_record.write("\t")
#			for q in p[0].tolist():
#				population_record.write(str(q)+",")
#			population_record.write("fitness,"+str(p.fitness.values)+",fitnesses,"+str(p.fitnesses.values)+"\n")
#			if p.fitness.values[0]>optimal_fitness:
#				for opt in optimal_solutions:
#					check_nonunique.append(all(elem in p[0][:-1] for elem in opt[0][:-1]))
#				if not any(check_nonunique):
#					optimal_solutions.append((p,current_generation))
#		population_record.write("SimulationGA,generation,"+str(current_generation)+"\n")
#		population_record.close()
#		end_time = datetime.datetime.now()
#		time_taken = end_time-start_time
#		opti = optimal_solutions[optimal_count:]
#		if len(opti)>0:
#			opt = open(working_directory+"optimal_solutions_discovered.csv","a")
#			for b in opti:
#				opt.write("SimulationGA,"+str(unique_run_seed)+",Solution_Parameters,"+str(b[0][0].tolist())+",Fitness,"+str(b[0].fitness.values)+",Discovered_Generation,"+str(b[1])+",Discovered_Time,"+str(end_time)+"\n")
#			opt.close()
#		optimal_count = len(optimal_solutions)
#
#	#Record GA results
#	if not os.path.exists(cwd+output_file):
#		results_file = open(cwd+output_file,"w")
#	results_file = open(cwd+results_file,"a")
#	results_file.write(str(logbook)+"\n")
#	results_file.close()
#	if not os.path.exists(cwd+"ga_times.csv"):
#		open(loc+"times.csv","w").close()
#	time = open(loc+"times.csv","a")
#	time.write("ga_seed,"+str(unique_run_seed)+",started_at,"+str(start_time)+",ended_at,"+str(end_time)+",total_time,"+str(time_taken)+"\n")
#	time.close()
#	return
#
#def create_individual(container):
#	global curr_pop, parameter_limits
#	new = [0]*(len(parameter_limits)+1)
#	for i in range(len(parameter_limits)):
#		if type(parameter_limits[i][0])==type(int()):
#			new[i] = int(random.uniform(parameter_limits[i][0], parameter_limits[i][1]))
#		else:
#			new[i] = round(random.uniform(parameter_limits[i][0], parameter_limits[i][1]),6)
#	new[-1] = curr_pop
#	curr_pop += 1
#	new = np.array(new, dtype=np.float64).reshape(1,-1)
#	return container(new)
#
#def favour_offspring(parents, offspring, MU):
#	choice = (list(zip(parents, [0]*len(parents))) +
#				list(zip(offspring, [1]*len(offspring))))
#	choice.sort(key=lambda x: ((x[0].fitness.values[0]), x[1]), reverse=True)
#	return [x[0] for x in choice[:MU]], [x[0] for x in choice[:MU] if x[1]==1]
#
#def log(logbook, population, gen, evals):
#	global statistics
#	record = statistics.compile(population) if statistics else {}
#	logbook.record(generation=gen,evaluations=evals,**record)
#	return
#
#def evaluate_population(population):
#	evaluated_population = population
#	return evaluated_population
#
##Define a function for crossover between 2 individuals (many are available in deap if individuals are in bitstring form)
#def mate(parent1, parent2):
#	global toolbox
#	child = toolbox.individual()
#	return child
#
##Define a function for mutating an individual (many are available in deap if individuals are in bitstring form)
#def mutate(individual):
#	global toolbox
#
#	return individual,
#
##Define a function for selecting parents (many are available in deap)
#def select_parents(individuals,k):
#	global toolbox
#	#Example selection function, randomly select 2 parents from population
#	#parents = [random.choice(individuals) for i in range(k)]
#	#return [toolbox.clone(ind) for ind in parents]
#
