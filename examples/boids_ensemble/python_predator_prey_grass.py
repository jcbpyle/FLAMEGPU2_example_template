from pyflamegpu import *
import sys, random, math
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import experiment_generator as exp
sns.set()

# Set whether to run single model or ensemble, agent population size, and simulation steps 
ENSEMBLE = True;
ENSEMBLE_RUNS = 2;
PREY_POPULATION_SIZE = 64;
PREDATOR_POPULATION_SIZE = 64;
GRASS_POPULATION_SIZE = 64;
#STEPS = 100;
STEPS = 10;
# Change to false if pyflamegpu has not been built with visualisation support
VISUALISATION = True;

"""
  FLAME GPU 2 implementation of the Predator, Prey and Grass model, using spatial3D messaging.
  This is based on the FLAME GPU 1 implementation, but with dynamic generation of agents. 
  The environment is wrapped as in FLAME GPU 1.
"""


"""
  Get the length of a vector
  @param x x component of the vector
  @param y y component of the vector
  @param z z component of the vector
  @return the length of the vector
"""
def vec3Length(x, y, z):
    return math.sqrt(x * x + y * y + z * z);

"""
  Add a scalar to a vector in-place
  @param x x component of the vector
  @param y y component of the vector
  @param z z component of the vector
  @param value scalar value to add
"""
def vec3Add(x, y, z, value):
    x += value;
    y += value;
    z += value;

"""
  Subtract a scalar from a vector in-place
  @param x x component of the vector
  @param y y component of the vector
  @param z z component of the vector
  @param value scalar value to subtract
"""
def vec3Sub(x, y, z, value):
    x -= value;
    y -= value;
    z -= value;

"""
  Multiply a vector by a scalar value in-place
  @param x x component of the vector
  @param y y component of the vector
  @param z z component of the vector
  @param multiplier scalar value to multiply by
"""
def vec3Mult(x, y, z, multiplier):
    x *= multiplier;
    y *= multiplier;
    z *= multiplier;

"""
  Divide a vector by a scalar value in-place
  @param x x component of the vector
  @param y y component of the vector
  @param z z component of the vector
  @param divisor scalar value to divide by
"""
def vec3Div(x, y, z, divisor):
    x /= divisor;
    y /= divisor;
    z /= divisor;


"""
  Normalize a 3 component vector in-place
  @param x x component of the vector
  @param y y component of the vector
  @param z z component of the vector
"""
def vec3Normalize(x, y, z):
    # Get the length
    length = vec3Length(x, y, z);
    vec3Div(x, y, z, length);

"""
  Clamp each component of a 3-part position to lie within a minimum and maximum value.
  Performs the operation in place
  Unlike the FLAME GPU 1 example, this is a clamping operation, rather than wrapping.
  @param x x component of the vector
  @param y y component of the vector
  @param z z component of the vector
  @param MIN_POSITION the minimum value for each component
  @param MAX_POSITION the maximum value for each component
"""
def clampPosition(x, y, z, MIN_POSITION, MAX_POSITION):
    x = MIN_POSITION if (x < MIN_POSITION) else x;
    x = MAX_POSITION if (x > MAX_POSITION) else x;

    y = MIN_POSITION if (y < MIN_POSITION) else y;
    y = MAX_POSITION if (y > MAX_POSITION) else y;

    z = MIN_POSITION if (z < MIN_POSITION) else z;
    z = MAX_POSITION if (z > MAX_POSITION) else z;

"""
  Ensure each component of a 3-part position lies within a minimum and maximum value, wrapping toroidally if bounds are exceeded. TODO: move away from wrapped edge same amount as bound crossed?
  Performs the operation in place
  @param x x component of the vector
  @param y y component of the vector
  @param z z component of the vector
  @param MIN_POSITION the minimum value for each component
  @param MAX_POSITION the maximum value for each component
"""
def boundPosition(x, y, z, MIN_POSITION, MAX_POSITION)
    agent_position.x = MAX_POSITION if (x<MIN_POSITION) else x;
    agent_position.x = MIN_POSITION if (x>MAX_POSITION) else x;

    agent_position.y = MAX_POSITION if (y<MIN_POSITION) else y;
    agent_position.y = MIN_POSITION if (y>MAX_POSITION) else y;

    agent_position.z = MAX_POSITION if (z<MIN_POSITION) else z;
    agent_position.z = MIN_POSITION if (z>MAX_POSITION) else z;

"""
  outputdata agent function for Prey agents, which outputs publicly visible properties to a message list
"""
prey_outputdata = """
FLAMEGPU_AGENT_FUNCTION(outputdata, MsgNone, MsgSpatial3D) {
    // Output each prey agent's location (and other visible properties if implemented)
    FLAMEGPU->prey_message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->prey_message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->prey_message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->prey_message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
    return ALIVE;
    }"""
"""
  inputdata agent function for Prey agents, which reads data from neighbouring agents, to perform predator avoidance, prey herding, and food seeking behaviour.
"""
prey_predator_location_inputdata = """
// Vector utility functions, see top of file for versions with commentary
FLAMEGPU_HOST_DEVICE_FUNCTION float vec3Length(const float x, const float y, const float z) {
    return sqrtf(x * x + y * y + z * z);
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Add(float &x, float &y, float &z, const float value) {
    x += value;
    y += value;
    z += value;
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Sub(float &x, float &y, float &z, const float value) {
    x -= value;
    y -= value;
    z -= value;
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Mult(float &x, float &y, float &z, const float multiplier) {
    x *= multiplier;
    y *= multiplier;
    z *= multiplier;
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Div(float &x, float &y, float &z, const float divisor) {
    x /= divisor;
    y /= divisor;
    z /= divisor;
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Normalize(float &x, float &y, float &z) {
    // Get the length
    float length = vec3Length(x, y, z);
    vec3Div(x, y, z, length);
}
FLAMEGPU_HOST_DEVICE_FUNCTION void clampPosition(float &x, float &y, float &z, const float MIN_POSITION, const float MAX_POSITION) {
    x = (x < MIN_POSITION)? MIN_POSITION: x;
    x = (x > MAX_POSITION)? MAX_POSITION: x;

    y = (y < MIN_POSITION)? MIN_POSITION: y;
    y = (y > MAX_POSITION)? MAX_POSITION: y;

    z = (z < MIN_POSITION)? MIN_POSITION: z;
    z = (z > MAX_POSITION)? MAX_POSITION: z;
}
// Agent function
FLAMEGPU_AGENT_FUNCTION(inputdata, MsgSpatial3D, MsgNone) {
    // Agent properties in local register
    int id = FLAMEGPU->getVariable<int>("id");
    // Agent position
    float agent_x = FLAMEGPU->getVariable<float>("x");
    float agent_y = FLAMEGPU->getVariable<float>("y");
    //Alter velocity to avoid predator agents
    float average_center_x = 0.0;
    float average_center_y = 0.0;
    float avoidance_x = 0.0;
    float avoidance_y = 0.0;
    float separation = 0.0;
    const float message_x = 0.0;
    const float message_y = 0.0;

    const float PRED_PREY_INTERACTION_RADIUS = FLAMEGPU->environment.getProperty<float>("PRED_PREY_INTERACTION_RADIUS");
    // Iterate location messages, accumulating relevant data and counts.
    for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
        
        message_x = message.getVariable<float>("x");
        message_y = message.getVariable<float>("y");

        // Check interaction radius
        separation = vec3Length(agent_x - message_x, agent_y - message_y, 0.0);

        if (separation < (PRED_PREY_INTERACTION_RADIUS) and separation>0.0) {
            // Update the percieved centre
            avoidance_x += PRED_PREY_INTERACTION_RADIUS / separation*(agent_x - message_x);
            avoidance_y += PRED_PREY_INTERACTION_RADIUS / separation*(agent_y - message_y);
        }
    }

    FLAMEGPU->setVariable<float>("steer_x", avoid_x);
    FLAMEGPU->setVariable<float>("steer_y", avoid_y);
    return ALIVE;
}
"""
prey_prey_location_inputdata = """
// Agent function
FLAMEGPU_AGENT_FUNCTION(inputdata, MsgSpatial3D, MsgNone) {
    // Agent properties in local register
    int id = FLAMEGPU->getVariable<int>("id");
    // Agent position
    float agent_x = FLAMEGPU->getVariable<float>("x");
    float agent_y = FLAMEGPU->getVariable<float>("y");
    //Alter velocity to herd together with fellow prey agents
    float group_center_x = 0.0;
    float group_center_y = 0.0;
    float group_velocity_x = 0.0;
    float group_velocity_y = 0.0;
    float avoidance_x = 0.0;
    float avoidance_y = 0.0;
    float separation = 0.0;
    int group_center_count = 0;
    const float message_x = 0.0;
    const float message_y = 0.0;
    const int message_id = 0;

    const float PREY_GROUP_COHESION_RADIUS = FLAMEGPU->environment.getProperty<float>("PREY_GROUP_COHESION_RADIUS");
    const float SAME_SPECIES_AVOIDANCE_RADIUS = FLAMEGPU->environment.getProperty<float>("SAME_SPECIES_AVOIDANCE_RADIUS");
    // Iterate location messages, accumulating relevant data and counts.
    for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
        
        message_x = message.getVariable<float>("x");
        message_y = message.getVariable<float>("y");
        message_id = message.getVariable<int>("id");

        // Check interaction radius
        separation = vec3Length(agent_x - message_x, agent_y - message_y, 0.0);

        if (separation < (PREY_GROUP_COHESION_RADIUS) and not id==message_id) {
            // Update the percieved centre
            group_center_x += message_x;
            group_center_y += message_y;
            group_center_count += 1;

            if (separation<(SAME_SPECIES_AVOIDANCE_RADIUS) and separation>0.0) {
                avoidance_x += SAME_SPECIES_AVOIDANCE_RADIUS/(separation*(agent_x-message_x));
                avoidance_y += SAME_SPECIES_AVOIDANCE_RADIUS/(separation*(agent_y-message_y));
            }
        }
    }

    if (group_centre_count>0) {
        group_center_x /= group_centre_count;
        group_center_y /= group_centre_count;
        group_velocity_x = (group_center_x - agent_x);
        group_velocity_y = (group_center_y - agent_y);
    }
    float current_steer_x = FLAMEGPU->getVariable<float>("steer_x");
    float current_steer_y = FLAMEGPU->getVariable<float>("steer_y");
    FLAMEGPU->setVariable<float>("steer_x", current_steer_x+avoid_x+group_velocity_x);
    FLAMEGPU->setVariable<float>("steer_y", current_steer_y+avoid_y+group_velocity_y);
    return ALIVE;
}
"""
prey_grass_location_input_data="""

"""


model = pyflamegpu.ModelDescription("Ensemble_Boids_BruteForce");


"""
  GLOBALS
"""
env = model.Environment();
# Population size to generate, if no agents are loaded from disk
env.newPropertyUInt("POPULATION_TO_GENERATE", POPULATION_SIZE);

# Number of steps to simulate
env.newPropertyUInt("STEPS", STEPS);

# Environment Bounds
env.newPropertyFloat("MIN_POSITION", -0.5);
env.newPropertyFloat("MAX_POSITION", +0.5);

# Initialisation parameter(s)
env.newPropertyFloat("MAX_INITIAL_SPEED", 1.0);
env.newPropertyFloat("MIN_INITIAL_SPEED", 0.01);

# Interaction radius
env.newPropertyFloat("INTERACTION_RADIUS", 0.1);
env.newPropertyFloat("SEPARATION_RADIUS", 0.005);

# Global Scalers
env.newPropertyFloat("TIME_SCALE", 0.0005);
env.newPropertyFloat("GLOBAL_SCALE", 0.15);

# Rule scalers
env.newPropertyFloat("STEER_SCALE", 0.65);
env.newPropertyFloat("COLLISION_SCALE", 0.75);
env.newPropertyFloat("MATCH_SCALE", 1.25);

"""
  Location message
"""
message = model.newMessageSpatial3D("location");
# Set the range and bounds.
message.setRadius(env.getPropertyFloat("INTERACTION_RADIUS"));
message.setMin(env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"));
message.setMax(env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"));
# A message to hold the location of an agent.
message.newVariableInt("id");
# X Y Z are implicit.
# message.newVariable<float>("x");
# message.newVariable<float>("y");
# message.newVariable<float>("z");
message.newVariableFloat("fx");
message.newVariableFloat("fy");
message.newVariableFloat("fz");
    
"""
  Boid agent
"""
agent = model.newAgent("Boid");
agent.newVariableInt("id");
agent.newVariableFloat("x");
agent.newVariableFloat("y");
agent.newVariableFloat("z");
agent.newVariableFloat("fx");
agent.newVariableFloat("fy");
agent.newVariableFloat("fz");
agent.newRTCFunction("outputdata", outputdata).setMessageOutput("location");
agent.newRTCFunction("inputdata", inputdata).setMessageInput("location");


# Add init function for creating boid population with random initial location
class initPopulation(pyflamegpu.HostFunctionCallback):
    def run(self,FLAMEGPU):
        populationSize = FLAMEGPU.environment.getPropertyUInt("POPULATION_TO_GENERATE");
        min_pos = FLAMEGPU.environment.getPropertyFloat("MIN_POSITION");
        max_pos = FLAMEGPU.environment.getPropertyFloat("MAX_POSITION");
        min_speed = FLAMEGPU.environment.getPropertyFloat("MIN_INITIAL_SPEED");
        max_speed = FLAMEGPU.environment.getPropertyFloat("MAX_INITIAL_SPEED");
        for i in range(populationSize):
            instance = FLAMEGPU.newAgent("Boid");
            instance.setVariableInt("id", i);
            instance.setVariableFloat("x", random.uniform(min_pos, max_pos));
            instance.setVariableFloat("y", random.uniform(min_pos, max_pos));
            instance.setVariableFloat("z", random.uniform(min_pos, max_pos));
            fx = random.uniform(-1, 1);
            fy = random.uniform(-1, 1);
            fz = random.uniform(-1, 1);
            fmagnitude = random.uniform(min_speed, max_speed);
            vec3Normalize(fx, fy, fz);
            vec3Mult(fx, fy, fz, fmagnitude);
            instance.setVariableFloat("fx", fx);
            instance.setVariableFloat("fy", fy);
            instance.setVariableFloat("fz", fz);
        return


# Add function callback to INIT function for population generation
initialPopulation = initPopulation();
model.addInitFunctionCallback(initialPopulation);


"""
  Control flow
"""    
# Layer #1
model.newLayer().addAgentFunction("Boid", "outputdata");
# Layer #2
model.newLayer().addAgentFunction("Boid", "inputdata");

"""
  Create Run Plan Vector
"""   
run_plan_vector = pyflamegpu.RunPlanVec(model, ENSEMBLE_RUNS);
run_plan_vector.setSteps(env.getPropertyUInt("STEPS"));
simulation_seed = random.randint(0,99999);
run_plan_vector.setRandomSimulationSeed(simulation_seed,1000);












test = exp.InitialStateGenerator();

test1 = test.setGlobalFloat("test_var",0);


test2 = exp.AgentPopulation("Boid");

test3 = test2.setPopSize(10);




boid_population = exp.AgentPopulation("Boid");
boid_population.setPopSizeRandom((256,1024));
boid_population.setVariableRandomPerAgent("x",(-1.0,1.0));
boid_population.setVariableRandomPerAgent("y",(-1.0,1.0));
boid_population.setVariableRandomPerAgent("z",(-1.0,1.0));
boid_population.setVariableRandomPerAgent("fx",(-1.0,1.0));
boid_population.setVariableRandomPerAgent("fy",(-1.0,1.0));
boid_population.setVariableRandomPerAgent("fz",(-1.0,1.0));


initial_states = exp.InitialStateGenerator();
initial_states.setGlobalRandom("test_global",(0,100));
initial_states.addAgentPopulation(boid_population);


experiment = exp.Experiment("test_experiment");
experiment.setModel(model);
experiment.initialStateGenerator(initial_states);
experiment.setSimulationSteps(10);
experiment.setRuns(3);


#experiment.begin();


print(experiment.generator.agent_list[0].name);



search = exp.Search();
search.GA();













# """
#   Create Model Runner
# """  
# if ENSEMBLE: 
#     simulation = pyflamegpu.CUDAEnsemble(model);
# else:
#     simulation = pyflamegpu.CUDASimulation(model);

# # Create and configure logging details 
# logging_config = pyflamegpu.LoggingConfig(model);
# agent_log = logging_config.agent("Boid");
# agent_log.logMeanFloat("x");
# agent_log.logMeanFloat("y");
# agent_log.logMeanFloat("z");
# agent_log.logMeanFloat("fx");
# agent_log.logMeanFloat("fy");
# agent_log.logMeanFloat("fz");
# # agent_log.logStandardDevFloat("x");
# # agent_log.logStandardDevFloat("y");
# # agent_log.logStandardDevFloat("z");
# agent_log.logStandardDevFloat("fx");
# agent_log.logStandardDevFloat("fy");
# agent_log.logStandardDevFloat("fz");
# step_log = pyflamegpu.StepLoggingConfig(logging_config);
# step_log.setFrequency(1);


# simulation.setStepLog(step_log);
# simulation.setExitLog(logging_config)


# """
#   Create Visualisation
# """
# if pyflamegpu.VISUALISATION and VISUALISATION and not ENSEMBLE:
#     visualisation = simulation.getVisualisation();
#     # Configure vis
#     envWidth = env.getPropertyFloat("MAX_POSITION") - env.getPropertyFloat("MIN_POSITION");
#     INIT_CAM = env.getPropertyFloat("MAX_POSITION") * 1.25;
#     visualisation.setInitialCameraLocation(INIT_CAM, INIT_CAM, INIT_CAM);
#     visualisation.setCameraSpeed(0.002 * envWidth);
#     circ_agt = visualisation.addAgent("Boid");
#     # Position vars are named x, y, z; so they are used by default
#     circ_agt.setModel(pyflamegpu.ICOSPHERE);
#     circ_agt.setModelScale(env.getPropertyFloat("SEPARATION_RADIUS"));

#     visualisation.activate();

# """
#   Initialise Model
# """
# simulation.initialise(sys.argv);

# """
#   Execution
# """
# if ENSEMBLE:
#     simulation.simulate(run_plan_vector);
# else:
#     simulation.simulate();

# """
#   Export Pop
# """
# # simulation.exportData("end.xml");

# # Join Visualisation
# if pyflamegpu.VISUALISATION and VISUALISATION and not ENSEMBLE:
#     visualisation.join();



# # Deal with logs
# if ENSEMBLE:
#     logs = simulation.getLogs();
# else:
#     logs = simulation.getRunLog();


# if ENSEMBLE:
#     positions_mean = [None]*ENSEMBLE_RUNS
#     positions_std = [None]*ENSEMBLE_RUNS
#     velocities_mean = [None]*ENSEMBLE_RUNS
#     velocities_std = [None]*ENSEMBLE_RUNS
#     text_pos = [None]*ENSEMBLE_RUNS
#     # Read logs
#     for i in range(len(logs)):
#         sl = logs[i].getStepLog();
#         positions_mean[i] = [[],[],[]];
#         velocities_mean[i] = [[],[],[]];
#         velocities_std[i] = [[],[],[]];
#         text_pos[i] = [[0,0,0,"start"],[0,0,0,"finish"]];
#         counter = 0;
#         for step in sl:
#             if counter>0:
#                 agents = step.getAgent("Boid");
#                 # Collect agent data
#                 positions_mean[i][0].append(agents.getMean("x"));
#                 positions_mean[i][1].append(agents.getMean("y"));
#                 positions_mean[i][2].append(agents.getMean("z"));
#                 velocities_mean[i][0].append(agents.getMean("fx"));
#                 velocities_mean[i][1].append(agents.getMean("fy"));
#                 velocities_mean[i][2].append(agents.getMean("fz"));
#                 velocities_std[i][0].append(agents.getStandardDev("fx"));
#                 velocities_std[i][1].append(agents.getStandardDev("fy"));
#                 velocities_std[i][2].append(agents.getStandardDev("fz"));
#                 #Set start and finish positions for graph text placement
#                 if counter==1:
#                     text_pos[i][0][0] = agents.getMean("x");
#                     text_pos[i][0][1] = agents.getMean("y");
#                     text_pos[i][0][2] = agents.getMean("z");
#                 elif counter==len(sl)-1:
#                     text_pos[i][1][0] = agents.getMean("x");
#                     text_pos[i][1][1] = agents.getMean("y");
#                     text_pos[i][1][2] = agents.getMean("z");
#             counter+=1;

#     # Generate graphs 
#     for j in range(ENSEMBLE_RUNS):
#         # Plot 3d graph of average flock position over simulation for individual model run
#         fig = plt.figure(figsize=(8,8));
#         ax = fig.gca(projection='3d');
#         ax.set_xlabel("Model environment x");
#         ax.set_ylabel("Model environment y");
#         ax.set_zlabel("Model environment z");
#         fig.suptitle("Ensemble run "+str(j)+" boids mean flock positions",fontsize=16);
#         label = "Boids mean flock position, ensemble run "+str(j);
#         fname = "average_flock_positions_run"+str(j)+".png";
#         # Position start and finish flock position text
#         for k in text_pos[j]:
#             ax.text(k[0],k[1],k[2],k[3],None);
#         ax.plot(positions_mean[j][0], positions_mean[j][1], positions_mean[j][2], label=label);
#         #ax.set_xlim3d([-1.0,1.0]);
#         #ax.set_ylim3d([-1.0,1.0]);
#         #ax.set_zlim3d([-1.0,1.0]);
#         ax.legend();
#         plt.savefig(fname,format='png');
#         plt.close(fig);

#         # Plot graphs for average of each fx, fy, and fz with standard deviation error bars
#         steplist = range(STEPS);
#         fig,(axx,axy,axz) = plt.subplots(1,3, figsize=(21,6));
#         fig.suptitle("Ensemble run "+str(j)+" mean boid velocities with std errorbar",fontsize=16);
#         velfname = "mean_velocities_run"+str(j)+".png";
#         axx.errorbar(steplist,velocities_mean[j][0],yerr=velocities_std[j][0],elinewidth=0.5,capsize=1.0);
#         axx.set_xlabel("Simulation step");
#         axx.set_ylabel("Boid agents average fx");
#         axy.errorbar(steplist,velocities_mean[j][1],yerr=velocities_std[j][1],elinewidth=0.5,capsize=1.0);
#         axy.set_xlabel("Simulation step");
#         axy.set_ylabel("Boid agents average fy");
#         axz.errorbar(steplist,velocities_mean[j][2],yerr=velocities_std[j][2],elinewidth=0.5,capsize=1.0);
#         axz.set_xlabel("Simulation step");
#         axz.set_ylabel("Boid agents average fz");
#         plt.savefig(velfname,format='png');
#         plt.close(fig);

#     # Plot every model in esemble's average flock position over simulation on the same 3d graph
#     fig = plt.figure(figsize=(12,12));
#     fig.suptitle("Ensemble Boids mean flock positions",fontsize=16);
#     ax = fig.gca(projection='3d');
#     ax.set_xlabel("Model environment x");
#     ax.set_ylabel("Model environment y");
#     ax.set_zlabel("Model environment z");
#     fname = "ensemble_average_flock_positions.png";
#     ## Plot start and finish text for each flock path ---VERY CLUTTERED---
#     # for i in text_pos:
#     #     for k in i:
#     #         ax.text(k[0],k[1],k[2],k[3],'x');
#     jcount = 0;
#     for j in positions_mean:
#         label1 = "Run "+str(jcount);
#         ax.plot(j[0], j[1], j[2], label=label1);
#         jcount+=1;
#     #ax.set_xlim3d([-1.0,1.0]);
#     #ax.set_ylim3d([-1.0,1.0]);
#     #ax.set_zlim3d([-1.0,1.0]);
#     ax.legend();
#     plt.savefig(fname,format='png');
#     plt.close(fig);
# else:
#     steps = logs.getStepLog();
#     for step in steps:
#         stepcount = step.getStepCount();
#         agents = step.getAgent("Boid");
#         print("Step: ",stepcount,"\tAverage flock position (x,y,z): ",agents.getMean("x"),", ",agents.getMean("y"),", ",agents.getMean("z"));

