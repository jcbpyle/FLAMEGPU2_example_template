import pytest
from unittest import TestCase
from pyflamegpu import *
import experiment_generator as exp

AGENT_NAME1 = 'Agent1'
AGENT_NAME2 = 'Agent2'
VARIABLE_NAME1 = 'Var1'
VARIABLE_NAME2 = 'Var2'
VARIABLE_NAME3 = 'Var3'
VARIABLE_NAME4 = 'Var4'
FUNCTION_NAME1 = 'Func1'
FUNCTION_NAME2 = 'Func2'
STATE_NAME1 = 'State1'
STATE_NAME2 = 'State2'

AP_MIN_POP1 = 25
AP_MAX_POP1 = 500
AP_MIN_POP2 = 125
AP_MAX_POP2 = 7500
AP_LIST = [10,25,100]
TEST_FLOAT1 = 7.25
TEST_FLOAT2 = 23.45
TEST_INT1 = 12
TEST_INT2 = 34
TEST_LIST1 = [1,3,8]
TEST_STRING1 = 'test_var_string'

EXPERIMENT_NAME1 = 'Experiment1'
EXPERIMENT_NAME2 = 'Experiment2'
EXPERIMENT_RUNS1 = 5
EXPERIMENT_RUNS2 = 50
EXPERIMENT_REPEATS1 = 3
EXPERIMENT_REPEATS2 = 8
EXPERIMENT_STEPS1 = 20
EXPERIMENT_STEPS2 = 75
EXPERIMENT_DIRECTORY = '/test_directory'
EXPERIMENT_FNAME = 'new_output.csv'

AGENT_COUNT = 100

class AgentPopulationTest(TestCase):

	def test_agent_population_init(self):
		ap = exp.AgentPopulation()
		assert ap.name=='agent'
		assert ap.agent_state=='DEFAULT1'
		assert ap.pop_min==1
		assert ap.pop_max==1024
		assert ap.pop_list==[]
		assert ap.variable_list==[]
		del ap

	def test_agent_population_setters(self):
		ap = exp.AgentPopulation()
		ap.setDefaultState(STATE_NAME2)
		assert ap.agent_state==STATE_NAME2
		ap.setPopSizeMin(AP_MIN_POP1)
		assert ap.pop_min==AP_MIN_POP1
		ap.setPopSizeMax(AP_MAX_POP1)
		assert ap.pop_max==AP_MAX_POP1
		ap.setPopSize((AP_MIN_POP2,AP_MAX_POP2))
		assert ap.pop_min==AP_MIN_POP2
		assert ap.pop_max==AP_MAX_POP2
		ap.setPopSizeList(AP_LIST)
		assert len(ap.pop_list)==len(AP_LIST)
		for i in range(len(AP_LIST)):
			assert ap.pop_list[i]==AP_LIST[i]
		ap.setPopSizeRandom((AP_MIN_POP1,AP_MAX_POP2))
		assert ap.pop_min>=AP_MIN_POP1
		assert ap.pop_max<=AP_MAX_POP2

		assert len(ap.variable_list)==0
		ap.setVariableFloat(VARIABLE_NAME1,TEST_FLOAT1)
		assert len(ap.variable_list)==1
		assert ap.variable_list[0]==(VARIABLE_NAME1,TEST_FLOAT1,True)
		ap.setVariableFloat(VARIABLE_NAME1,(TEST_FLOAT1,TEST_FLOAT2))
		assert len(ap.variable_list)==1
		assert not ap.variable_list[0]==(VARIABLE_NAME1,TEST_FLOAT1,True)
		assert ap.variable_list[0][0]==VARIABLE_NAME1
		assert ap.variable_list[0][1]>=TEST_FLOAT1
		assert ap.variable_list[0][1]<=TEST_FLOAT2
		ap.setVariableFloat(VARIABLE_NAME2,TEST_FLOAT2)
		assert len(ap.variable_list)==2
		assert ap.variable_list[1]==(VARIABLE_NAME2,TEST_FLOAT2,True)


		ap.setVariableInt(VARIABLE_NAME1,TEST_INT1)
		assert len(ap.variable_list)==2
		assert ap.variable_list[0]==(VARIABLE_NAME1,TEST_INT1,True)
		ap.setVariableInt(VARIABLE_NAME1,(TEST_INT1,TEST_INT2))
		assert len(ap.variable_list)==2
		assert ap.variable_list[0][0]==VARIABLE_NAME1
		assert ap.variable_list[0][1]>=TEST_INT1
		assert ap.variable_list[0][1]<=TEST_INT2

		ap.setVariableList(VARIABLE_NAME1,TEST_LIST1)
		assert len(ap.variable_list)==2
		assert ap.variable_list[0]==(VARIABLE_NAME1,TEST_LIST1,True)

		ap.setVariableList(VARIABLE_NAME1,TEST_STRING1)
		assert len(ap.variable_list)==2
		assert ap.variable_list[0]==(VARIABLE_NAME1,TEST_STRING1,True)

		assert len(ap.variable_list)==2
		ap.setVariableRandomPerAgent(VARIABLE_NAME1,(TEST_FLOAT1,TEST_FLOAT2))
		assert ap.variable_list[0]==(VARIABLE_NAME1,(TEST_FLOAT1,TEST_FLOAT2),'random.uniform',False)
		ap.setVariableRandomPerAgent(VARIABLE_NAME1,(TEST_INT1,TEST_INT2),distribution='random.randint')
		assert ap.variable_list[0]==(VARIABLE_NAME1,(TEST_INT1,TEST_INT2),'random.randint',False)
		del ap


class InitialStateGeneratorTest(TestCase):
	
	def test_initial_state_generation(self):
		isg1 = exp.InitialStateGenerator();
		assert isg1.file==None
		assert isg1.global_list==[]
		assert isg1.agent_list==[]

		isg1.initialStateFile('newfile.csv')
		assert isg1.file=='newfile.csv'
		isg1.initialStateFile(0)
		assert isg1.file==None
		isg1.initialStateFile(None)
		assert isg1.file==None

		assert len(isg1.global_list)==0
		isg1.setGlobalFloat(VARIABLE_NAME1,TEST_FLOAT1)
		assert len(isg1.global_list)==1
		assert isg1.global_list[0]==(VARIABLE_NAME1,TEST_FLOAT1)
		isg1.setGlobalFloat(VARIABLE_NAME1,(TEST_FLOAT1,TEST_FLOAT2))
		assert len(isg1.global_list)==1
		assert isg1.global_list[0][0]==VARIABLE_NAME1
		assert isg1.global_list[0][1]>=TEST_FLOAT1
		assert isg1.global_list[0][1]<=TEST_FLOAT2


		isg1.setGlobalInt(VARIABLE_NAME2,TEST_INT1)
		assert len(isg1.global_list)==2
		assert isg1.global_list[1]==(VARIABLE_NAME2,TEST_INT1)
		isg1.setGlobalInt(VARIABLE_NAME2,(TEST_INT1,TEST_INT2))
		assert len(isg1.global_list)==2
		assert isg1.global_list[1][0]==VARIABLE_NAME2
		assert isg1.global_list[1][1]>=TEST_INT1
		assert isg1.global_list[1][1]<=TEST_INT2

		isg1.setGlobalList(VARIABLE_NAME2,TEST_LIST1)
		assert len(isg1.global_list)==2
		assert isg1.global_list[1]==(VARIABLE_NAME2,TEST_LIST1)

		isg1.setGlobalList(VARIABLE_NAME2,TEST_STRING1)
		assert len(isg1.global_list)==2
		assert isg1.global_list[1]==(VARIABLE_NAME2,TEST_STRING1)

		isg1.setGlobalRandom(VARIABLE_NAME2,(TEST_FLOAT1,TEST_FLOAT2))
		assert len(isg1.global_list)==2
		assert isg1.global_list[1][0]==VARIABLE_NAME2
		assert isg1.global_list[1][1]>=TEST_FLOAT1
		assert isg1.global_list[1][1]<=TEST_FLOAT2

		ap1 = exp.AgentPopulation(AGENT_NAME1)
		ap2 = exp.AgentPopulation(AGENT_NAME2)
		assert len(isg1.agent_list)==0
		isg1.addAgentPopulation(ap1)
		assert len(isg1.agent_list)==1
		assert isg1.agent_list[0]==ap1
		assert isg1.agent_list[0].name==ap1.name
		isg1.addAgentPopulation(ap2)
		assert len(isg1.agent_list)==2
		assert isg1.agent_list[1]==ap2
		assert isg1.agent_list[1].name==ap2.name
		del isg1


class ExperimentTest(TestCase):
	agent_fn1 = '''
	FLAMEGPU_AGENT_FUNCTION(agent_fn1, MsgNone, MsgNone) {
	// do nothing
	return ALIVE;
	}
	'''
	agent_fn2 = '''
	FLAMEGPU_AGENT_FUNCTION(agent_fn2, MsgNone, MsgNone) {
	// do nothing
	return ALIVE;
	}
	'''
	
	def test_experiment_init(self):
		ex1 = exp.Experiment();
		assert ex1.name=='experiment'
		assert ex1.runs==1
		assert ex1.repeats==1
		assert ex1.steps==10
		assert ex1.output_location==''
		assert ex1.filename=='output.txt'
		assert ex1.model==''
		assert ex1.generator==None
		del ex1

		ex1 = exp.Experiment(EXPERIMENT_NAME1);
		assert ex1.name==EXPERIMENT_NAME1
		assert ex1.runs==1
		assert ex1.repeats==1
		assert ex1.steps==10
		assert ex1.output_location==''
		assert ex1.filename=='output.txt'
		assert ex1.model==''
		assert ex1.generator==None
		del ex1

		ex1 = exp.Experiment(EXPERIMENT_NAME1, EXPERIMENT_RUNS2, EXPERIMENT_STEPS2);
		assert ex1.name==EXPERIMENT_NAME1
		assert ex1.runs==EXPERIMENT_RUNS2
		assert ex1.repeats==1
		assert ex1.steps==EXPERIMENT_STEPS2
		assert ex1.output_location==''
		assert ex1.filename=='output.txt'
		assert ex1.model==''
		assert ex1.generator==None
		del ex1

		ex1 = exp.Experiment(EXPERIMENT_NAME1, EXPERIMENT_RUNS2, EXPERIMENT_STEPS2, EXPERIMENT_REPEATS2);
		assert ex1.name==EXPERIMENT_NAME1
		assert ex1.runs==EXPERIMENT_RUNS2
		assert ex1.repeats==EXPERIMENT_REPEATS2
		assert ex1.steps==EXPERIMENT_STEPS2
		assert ex1.output_location==''
		assert ex1.filename=='output.txt'
		assert ex1.model==''
		assert ex1.generator==None
		del ex1

		ex1 = exp.Experiment(name=EXPERIMENT_NAME2, runs=EXPERIMENT_RUNS1, steps=EXPERIMENT_STEPS1, repeats=EXPERIMENT_REPEATS2);
		assert ex1.name==EXPERIMENT_NAME2
		assert ex1.runs==EXPERIMENT_RUNS1
		assert ex1.repeats==EXPERIMENT_REPEATS2
		assert ex1.steps==EXPERIMENT_STEPS1
		assert ex1.output_location==''
		assert ex1.filename=='output.txt'
		assert ex1.model==''
		assert ex1.generator==None
		del ex1

	def test_experiment_setters(self):
		ex1 = exp.Experiment();
		ex1.setRuns(EXPERIMENT_RUNS1)
		assert ex1.runs==EXPERIMENT_RUNS1
		ex1.setRepeats(EXPERIMENT_REPEATS1)
		assert ex1.repeats==EXPERIMENT_REPEATS1
		ex1.setSimulationSteps(EXPERIMENT_STEPS1)
		assert ex1.steps==EXPERIMENT_STEPS1
		ex1.setModelLogDirectory(EXPERIMENT_DIRECTORY)
		assert ex1.output_location==EXPERIMENT_DIRECTORY
		ex1.setModelLogFile(EXPERIMENT_FNAME)
		assert ex1.filename==EXPERIMENT_FNAME

		m = pyflamegpu.ModelDescription('model1')
		a = m.newAgent(AGENT_NAME1)
		ex1.setModel(m)
		assert ex1.model==m
		assert ex1.model.getName()=='model1'
		del ex1

	def test_experiment(self):
		ex1 = exp.Experiment();
		ex1.setSimulationSteps(1)
		m = pyflamegpu.ModelDescription('model1')
		a = m.newAgent(AGENT_NAME1)
		a.newRTCFunction("agent_fn1", self.agent_fn1);
		m.newLayer("Layer1").addAgentFunction(AGENT_NAME1, "agent_fn1");
		ex1.setModel(m)
		ex1.begin()

		ex1.setRuns(EXPERIMENT_RUNS1)
		ex1.begin()
		del ex1

test_ap = AgentPopulationTest()
test_ap.test_agent_population_init()
test_ap.test_agent_population_setters()

test_isg = InitialStateGeneratorTest()
test_isg.test_initial_state_generation()

test_ex = ExperimentTest()
test_ex.test_experiment_init()
test_ex.test_experiment_setters()
test_ex.test_experiment()