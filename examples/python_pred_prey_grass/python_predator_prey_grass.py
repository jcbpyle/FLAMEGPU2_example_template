from pyflamegpu import *
import sys, random, math
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import experiment_generator as exp
sns.set()

# Set whether to run single model or ensemble, agent population size, and simulation steps 
ENSEMBLE = False;
ENSEMBLE_RUNS = 3;
PREY_POPULATION_SIZE = 64;
PREDATOR_POPULATION_SIZE = 10;
GRASS_POPULATION_SIZE = 200;
#STEPS = 100;
STEPS = 40;
# Change to false if pyflamegpu has not been built with visualisation support
VISUALISATION = False;

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
  Get the length of a vector
  @param x x component of the vector
  @param y y component of the vector
  @return the length of the vector
"""
def vec2Length(x, y):
    return math.sqrt(x * x + y * y);


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
  Divide a vector by a scalar value in-place
  @param x x component of the vector
  @param y y component of the vector
  @param divisor scalar value to divide by
"""
def vec2Div(x, y, divisor):
    x /= divisor;
    y /= divisor;


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
  Normalize a 2 component vector in-place
  @param x x component of the vector
  @param y y component of the vector
"""
def vec2Normalize(x, y):
    # Get the length
    length = vec2Length(x, y);
    vec2Div(x, y, length);

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
  Ensure each component of a 2-part position lies within a minimum and maximum value, wrapping toroidally if bounds are exceeded. TODO: move away from wrapped edge same amount as bound crossed?
  Performs the operation in place
  @param x x component of the vector
  @param y y component of the vector
  @param MIN_POSITION the minimum value for each component
  @param MAX_POSITION the maximum value for each component
"""
def boundPosition(x, y, MIN_POSITION, MAX_POSITION):
    agent_position.x = MAX_POSITION if (x<MIN_POSITION) else x;
    agent_position.x = MIN_POSITION if (x>MAX_POSITION) else x;

    agent_position.y = MAX_POSITION if (y<MIN_POSITION) else y;
    agent_position.y = MIN_POSITION if (y>MAX_POSITION) else y;

"""
  PREY
  prey_output_location_data agent function for Prey agents, which outputs publicly visible properties to a message list
"""
prey_output_location_data = """
/*// Vector utility functions, see top of file for versions with commentary
FLAMEGPU_HOST_DEVICE_FUNCTION float vec3Length(const float x, const float y, const float z) {
  return sqrtf(x * x + y * y + z * z);
}
FLAMEGPU_HOST_DEVICE_FUNCTION float vec2Length(const float x, const float y) {
  return sqrtf(x * x + y * y);
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
FLAMEGPU_HOST_DEVICE_FUNCTION void vec2Div(float &x, float &y, const float divisor) {
  x /= divisor;
  y /= divisor;
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Normalize(float &x, float &y, float &z) {
  // Get the length
  float length = vec3Length(x, y, z);
  vec3Div(x, y, z, length);
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec2Normalize(float &x, float &y) {
  // Get the length
  float length = vec2Length(x, y);
  vec2Div(x, y, length);
}
FLAMEGPU_HOST_DEVICE_FUNCTION void clampPosition(float &x, float &y, float &z, const float MIN_POSITION, const float MAX_POSITION) {
    x = (x < MIN_POSITION)? MIN_POSITION: x;
    x = (x > MAX_POSITION)? MAX_POSITION: x;

    y = (y < MIN_POSITION)? MIN_POSITION: y;
    y = (y > MAX_POSITION)? MAX_POSITION: y;

    z = (z < MIN_POSITION)? MIN_POSITION: z;
    z = (z > MAX_POSITION)? MAX_POSITION: z;
}
FLAMEGPU_HOST_DEVICE_FUNCTION void boundPosition(float &x, float &y, const float MIN_POSITION, const float MAX_POSITION) {
  x = (x < MIN_POSITION)? MAX_POSITION: x;
  x = (x > MAX_POSITION)? MIN_POSITION: x;

  y = (y < MIN_POSITION)? MAX_POSITION: y;
  y = (y > MAX_POSITION)? MIN_POSITION: y;
}*/
FLAMEGPU_AGENT_FUNCTION(prey_output_location_data, MsgNone, MsgSpatial2D) {
  // Output each prey agent's location (and other visible properties if implemented)
  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  return ALIVE;
  }
"""
"""
  prey_avoid_predators agent function for Prey agents, which reads data from neighbouring predator agents to alter course as to avoid them
"""
prey_avoid_predators = """
FLAMEGPU_HOST_DEVICE_FUNCTION float vec2Length(const float x, const float y) {
  return sqrtf(x * x + y * y);
}
// Agent function
FLAMEGPU_AGENT_FUNCTION(prey_avoid_predators, MsgSpatial2D, MsgNone) {
  // Agent properties in local register
  int id = FLAMEGPU->getVariable<int>("id");
  // Agent position
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  //Alter velocity to avoid predator agents
  float avoid_fx = 0.0;
  float avoid_fy = 0.0;
  float separation = 0.0;
  float message_x = 0.0;
  float message_y = 0.0;
  const float PRED_PREY_INTERACTION_RADIUS = FLAMEGPU->environment.getProperty<float>("PRED_PREY_INTERACTION_RADIUS");

  // Iterate location messages, accumulating relevant data and counts.
  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y)) {
      
    message_x = message.getVariable<float>("x");
    message_y = message.getVariable<float>("y");

    // Check interaction radius
    separation = vec2Length(agent_x - message_x, agent_y - message_y);

    if (separation < (PRED_PREY_INTERACTION_RADIUS) and separation>0.0) {
      // Update the percieved centre
      avoid_fx += PRED_PREY_INTERACTION_RADIUS / separation*(agent_x - message_x);
      avoid_fy += PRED_PREY_INTERACTION_RADIUS / separation*(agent_y - message_y);
    }
  }

  FLAMEGPU->setVariable<float>("steer_x", avoid_fx);
  FLAMEGPU->setVariable<float>("steer_y", avoid_fy);
  return ALIVE;
}
"""
"""
  prey_flock agent function for Prey agents, which reads data from neighbouring prey agents to alter course to produce flocking behaviour
"""
prey_flock = """
FLAMEGPU_HOST_DEVICE_FUNCTION float vec2Length(const float x, const float y) {
  return sqrtf(x * x + y * y);
}
// Agent function
FLAMEGPU_AGENT_FUNCTION(prey_flock, MsgSpatial2D, MsgNone) {
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
  float avoid_fx = 0.0;
  float avoid_fy = 0.0;
  float separation = 0.0;
  int group_center_count = 0;
  float message_x = 0.0;
  float message_y = 0.0;
  int message_id = 0;
  const float PREY_GROUP_COHESION_RADIUS = FLAMEGPU->environment.getProperty<float>("PREY_GROUP_COHESION_RADIUS");
  const float SAME_SPECIES_AVOIDANCE_RADIUS = FLAMEGPU->environment.getProperty<float>("SAME_SPECIES_AVOIDANCE_RADIUS");

  // Iterate location messages, accumulating relevant data and counts.
  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y)) {
      
    message_x = message.getVariable<float>("x");
    message_y = message.getVariable<float>("y");
    message_id = message.getVariable<int>("id");

    // Check interaction radius
    separation = vec2Length(agent_x - message_x, agent_y - message_y);

    if (separation < (PREY_GROUP_COHESION_RADIUS) and not id==message_id) {
      // Update the percieved centre
      group_center_x += message_x;
      group_center_y += message_y;
      group_center_count += 1;

      if (separation<(SAME_SPECIES_AVOIDANCE_RADIUS) and separation>0.0) {
        avoid_fx += SAME_SPECIES_AVOIDANCE_RADIUS/(separation*(agent_x-message_x));
        avoid_fy += SAME_SPECIES_AVOIDANCE_RADIUS/(separation*(agent_y-message_y));
      }
    }
  }

  if (group_center_count>0) {
    group_center_x /= group_center_count;
    group_center_y /= group_center_count;
    group_velocity_x = (group_center_x - agent_x);
    group_velocity_y = (group_center_y - agent_y);
  }
  float current_steer_x = FLAMEGPU->getVariable<float>("steer_x");
  float current_steer_y = FLAMEGPU->getVariable<float>("steer_y");
  FLAMEGPU->setVariable<float>("steer_x", current_steer_x+avoid_fx+group_velocity_x);
  FLAMEGPU->setVariable<float>("steer_y", current_steer_y+avoid_fy+group_velocity_y);
  return ALIVE;
}
"""
"""
  prey_move agent function for Prey agents, which uses combined steering away from predators and towards prey to alter course and moves based on agent speed and time step
"""
prey_move="""
FLAMEGPU_HOST_DEVICE_FUNCTION float vec2Length(const float x, const float y) {
  return sqrtf(x * x + y * y);
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec2Div(float &x, float &y, const float divisor) {
  x /= divisor;
  y /= divisor;
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec2Normalize(float &x, float &y) {
  // Get the length
  float length = vec2Length(x, y);
  vec2Div(x, y, length);
}
FLAMEGPU_HOST_DEVICE_FUNCTION void boundPosition(float &x, float &y, const float MIN_POSITION, const float MAX_POSITION) {
  x = (x < MIN_POSITION)? MAX_POSITION: x;
  x = (x > MAX_POSITION)? MIN_POSITION: x;

  y = (y < MIN_POSITION)? MAX_POSITION: y;
  y = (y > MAX_POSITION)? MIN_POSITION: y;
}
FLAMEGPU_AGENT_FUNCTION(prey_move, MsgNone, MsgNone) {
  //Agent position vector
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_fx = FLAMEGPU->getVariable<float>("fx");
  float agent_fy = FLAMEGPU->getVariable<float>("fy");
  float agent_steerx = FLAMEGPU->getVariable<float>("steer_x");
  float agent_steery = FLAMEGPU->getVariable<float>("steer_y");

  //Adjust the velocity according to the steering velocity
  agent_fx += agent_steerx;
  agent_fy += agent_steery;

  //Limit the speed of the avoidance velocity
  float current_speed = vec2Length(agent_fx, agent_fy);
  if (current_speed > 1.0) {
    vec2Normalize(agent_fx, agent_fy);
  }

  //Integrate the position by applying moving according to the velocity
  const float DELTA_TIME = FLAMEGPU->environment.getProperty<float>("DELTA_TIME");
  agent_x += agent_fx * DELTA_TIME;
  agent_y += agent_fy * DELTA_TIME;


  //Bound the position within the environment 
  boundPosition(agent_x, agent_y, FLAMEGPU->environment.getProperty<float>("MIN_POSITION"), FLAMEGPU->environment.getProperty<float>("MAX_POSITION"));

  //Update the agents position and velocity
  FLAMEGPU->setVariable<float>("x",agent_x);
  FLAMEGPU->setVariable<float>("y",agent_y);
  FLAMEGPU->setVariable<float>("fx",agent_fx);
  FLAMEGPU->setVariable<float>("fy",agent_fy);

  //reduce life by one unit of energy
  FLAMEGPU->setVariable<int>("life",FLAMEGPU->getVariable<int>("life")-1);

  return ALIVE;
}
"""
"""
  prey_eaten agent function for Prey agents, which checks if nearby predators are within killing distance and messages the closest to indicate a succesful kill. The prey agent dies in this case
"""
prey_eaten = """
FLAMEGPU_HOST_DEVICE_FUNCTION float vec2Length(const float x, const float y) {
  return sqrtf(x * x + y * y);
}
FLAMEGPU_AGENT_FUNCTION(prey_eaten, MsgSpatial2D, MsgSpatial2D) {
  int eaten = 0;
  int predator_id = -1;
  float closest_predator = FLAMEGPU->environment.getProperty<float>("PREDATOR_KILL_DISTANCE");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float message_x = 0.0;
  float message_y = 0.0;
  int message_id = 0;
  float distance = 0.0;

  // Iterate location messages, accumulating relevant data and counts.
  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y)) {
    message_x = message.getVariable<float>("x");
    message_y = message.getVariable<float>("y");
    message_id = message.getVariable<int>("id");
    distance = vec2Length(agent_x - message_x, agent_y - message_y);
    if (distance < closest_predator) {
      predator_id = message_id;
      closest_predator = distance;
      eaten = 1;
    }
  }

  //if one or more predators were within killing distance then notify the nearest predator that it has eaten this prey via a prey eaten message.
  if (eaten) {
    FLAMEGPU->message_out.setVariable<int>("predator_id", predator_id);
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    return DEAD;
  }

  //return eaten value to remove dead (eaten == 1) agents from the simulation
  return ALIVE;
}
"""
"""
  prey_eat_or_starve agent function for Prey agents, which checks if any grass agents have sent an eaten message and increases its own life per grass eaten message recieved with matching id
"""
prey_eat_or_starve = """
FLAMEGPU_AGENT_FUNCTION(prey_eat_or_starve, MsgSpatial2D, MsgNone) {
  int agent_life = FLAMEGPU->getVariable<int>("life");
  int agent_id = FLAMEGPU->getVariable<int>("id");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  int message_id = 0;
  const int GAIN_FROM_FOOD_PREY = FLAMEGPU->environment.getProperty<int>("GAIN_FROM_FOOD_PREY");
  // Iterate grass eaten messages, accumulating relevant data and counts.
  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y)) {
    message_id = message.getVariable<int>("prey_id");
    if (agent_id == message_id) {
      agent_life += GAIN_FROM_FOOD_PREY;
    }
  }

  //if one or more predators were within killing distance then notify the nearest predator that it has eaten this prey via a prey eaten message.
  if (agent_life < 1) {
    return DEAD;
  }
  else {
   FLAMEGPU->setVariable<int>("life", agent_life);
  }

  //return value to remove dead agents from the simulation
  return ALIVE;
}
"""
"""
  prey_reproduce agent function for Prey agents, which uses the reproduction chance environment variable and a random number to possibly generate a new prey agent in a random location. 
    The parent agent's life value is halved and shared with the offspring prey agent.
"""
prey_reproduce="""
FLAMEGPU_AGENT_FUNCTION(prey_reproduce, MsgNone, MsgNone) {
  float reproduction = FLAMEGPU->random.uniform<float>();
  const float PREY_REPRODUCTION_CHANCE = FLAMEGPU->environment.getProperty<float>("PREY_REPRODUCTION_CHANCE");
  if (reproduction<PREY_REPRODUCTION_CHANCE) {
    const float BOUNDS_WIDTH = FLAMEGPU->environment.getProperty<float>("BOUNDS_WIDTH");
    const float SPEED_DIFFERENCE = FLAMEGPU->environment.getProperty<float>("SPEED_DIFFERENCE");
    int new_id = FLAMEGPU->getVariable<int>("id") * 3;
    FLAMEGPU->agent_out.setVariable<int>("id", new_id);
    reproduction = (FLAMEGPU->random.uniform<float>()*BOUNDS_WIDTH)-(BOUNDS_WIDTH/2.0);
    FLAMEGPU->agent_out.setVariable<float>("x", reproduction);
    reproduction = (FLAMEGPU->random.uniform<float>()*BOUNDS_WIDTH)-(BOUNDS_WIDTH/2.0);
    FLAMEGPU->agent_out.setVariable<float>("y", reproduction);
    reproduction = (FLAMEGPU->random.uniform<float>()*SPEED_DIFFERENCE)-(SPEED_DIFFERENCE/2.0);
    FLAMEGPU->agent_out.setVariable<float>("fx", reproduction);
    reproduction = (FLAMEGPU->random.uniform<float>()*SPEED_DIFFERENCE)-(SPEED_DIFFERENCE/2.0);
    FLAMEGPU->agent_out.setVariable<float>("fy", reproduction);
    FLAMEGPU->agent_out.setVariable<float>("type", FLAMEGPU->getVariable<float>("type"));
    FLAMEGPU->agent_out.setVariable<float>("steer_x", 0.0);
    FLAMEGPU->agent_out.setVariable<float>("steer_y", 0.0);
    int half_life = (int)(FLAMEGPU->getVariable<int>("life")/2.0);
    FLAMEGPU->agent_out.setVariable<int>("life", half_life);
    FLAMEGPU->setVariable<int>("life", half_life);
  }
  return ALIVE;
}
"""
"""
  PREDATOR
  predator_output_location_data agent function for predator agents, which outputs publicly visible properties to a message list
"""
predator_output_location_data = """
FLAMEGPU_AGENT_FUNCTION(predator_output_location_data, MsgNone, MsgSpatial2D) {
  // Output each predator agent's location (and other visible properties if implemented)
  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  return ALIVE;
  }
"""
"""
  predator_follow_prey agent function for predator agents, which reads data from neighbouring prey agents to alter course in pursuit
"""
predator_follow_prey = """
FLAMEGPU_HOST_DEVICE_FUNCTION float vec2Length(const float x, const float y) {
  return sqrtf(x * x + y * y);
}
// Agent function
FLAMEGPU_AGENT_FUNCTION(predator_follow_prey, MsgSpatial2D, MsgNone) {
  // Agent properties in local register
  int id = FLAMEGPU->getVariable<int>("id");
  // Agent position
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_steerx = FLAMEGPU->getVariable<float>("steer_x");
  float agent_steery = FLAMEGPU->getVariable<float>("steer_y");
  //Alter velocity to pursue prey agents
  float closest_prey_distance = FLAMEGPU->environment.getProperty<float>("PRED_PREY_INTERACTION_RADIUS");
  int can_see_prey = 0;
  float closest_prey_x = 0.0;
  float closest_prey_y = 0.0;
  float message_x = 0.0;
  float message_y = 0.0;
  float separation = 0.0;

  // Iterate location messages, accumulating relevant data and counts.
  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y)) {
    message_x = message.getVariable<float>("x");
    message_y = message.getVariable<float>("y");

    // Check interaction radius
    separation = vec2Length(agent_x - message_x, agent_y - message_y);

    if (separation < closest_prey_distance) {
      // Update the percieved closest prey agent
      closest_prey_x = message_x;
      closest_prey_y = message_y;
      closest_prey_distance = separation;
      can_see_prey = 1;
    }
  }

  if (can_see_prey) {
    agent_steerx = closest_prey_x - agent_x;
    agent_steery = closest_prey_y - agent_y;
  }
  FLAMEGPU->setVariable<float>("steer_x", agent_steerx);
  FLAMEGPU->setVariable<float>("steer_y", agent_steery);
  return ALIVE;
}
"""
"""
  predator_avoidance agent function for Predator agents, which uses other predator location data to affect steering away, reproducing simple territorial behaviour
"""
predator_avoidance = """
FLAMEGPU_HOST_DEVICE_FUNCTION float vec2Length(const float x, const float y) {
  return sqrtf(x * x + y * y);
}
// Agent function
FLAMEGPU_AGENT_FUNCTION(predator_avoidance, MsgSpatial2D, MsgNone) {
  // Agent properties in local register
  int id = FLAMEGPU->getVariable<int>("id");
  // Agent position
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  //Alter velocity to avoid fellow predator agents
  float avoidance_x = 0.0;
  float avoidance_y = 0.0;
  float separation = 0.0;
  float message_x = 0.0;
  float message_y = 0.0;
  int message_id = 0;
  const float SAME_SPECIES_AVOIDANCE_RADIUS = FLAMEGPU->environment.getProperty<float>("SAME_SPECIES_AVOIDANCE_RADIUS");

  // Iterate location messages, accumulating relevant data and counts.
  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y)) {
      
    message_x = message.getVariable<float>("x");
    message_y = message.getVariable<float>("y");
    message_id = message.getVariable<int>("id");

    // Check interaction radius
    separation = vec2Length(agent_x - message_x, agent_y - message_y);

    if (separation < (SAME_SPECIES_AVOIDANCE_RADIUS) and separation>0.0 and not id==message_id) {
      avoidance_x += SAME_SPECIES_AVOIDANCE_RADIUS/(separation*(agent_x-message_x));
      avoidance_y += SAME_SPECIES_AVOIDANCE_RADIUS/(separation*(agent_y-message_y));
    }
  }

  float current_steer_x = FLAMEGPU->getVariable<float>("steer_x");
  float current_steer_y = FLAMEGPU->getVariable<float>("steer_y");
  FLAMEGPU->setVariable<float>("steer_x", current_steer_x+avoidance_x);
  FLAMEGPU->setVariable<float>("steer_y", current_steer_y+avoidance_y);
  return ALIVE;
}
"""
"""
  predator_move agent function for Predator agents, which uses steering away from predators (predator_avoidance) and prey pursuit to alter course.
   The environment properties of predator speed advantage and time step delta are applied to alter its location.
"""
predator_move="""
FLAMEGPU_HOST_DEVICE_FUNCTION float vec2Length(const float x, const float y) {
  return sqrtf(x * x + y * y);
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec2Div(float &x, float &y, const float divisor) {
  x /= divisor;
  y /= divisor;
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec2Normalize(float &x, float &y) {
  // Get the length
  float length = vec2Length(x, y);
  vec2Div(x, y, length);
}
FLAMEGPU_HOST_DEVICE_FUNCTION void boundPosition(float &x, float &y, const float MIN_POSITION, const float MAX_POSITION) {
  x = (x < MIN_POSITION)? MAX_POSITION: x;
  x = (x > MAX_POSITION)? MIN_POSITION: x;

  y = (y < MIN_POSITION)? MAX_POSITION: y;
  y = (y > MAX_POSITION)? MIN_POSITION: y;
}
FLAMEGPU_AGENT_FUNCTION(predator_move, MsgNone, MsgNone) {
  //Agent position vector
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_fx = FLAMEGPU->getVariable<float>("fx");
  float agent_fy = FLAMEGPU->getVariable<float>("fy");
  float agent_steerx = FLAMEGPU->getVariable<float>("steer_x");
  float agent_steery = FLAMEGPU->getVariable<float>("steer_y");

  //Adjust the velocity according to the steering velocity
  agent_fx += agent_steerx;
  agent_fy += agent_steery;

  //Limit the speed of the avoidance velocity
  float current_speed = vec2Length(agent_fx, agent_fy);
  if (current_speed > 1.0) {
    vec2Normalize(agent_fx, agent_fy);
  }

  //Integrate the position by applying moving according to the velocity
  const float DELTA_TIME = FLAMEGPU->environment.getProperty<float>("DELTA_TIME");
  const float PRED_SPEED_ADVANTAGE = FLAMEGPU->environment.getProperty<float>("PRED_SPEED_ADVANTAGE");
  agent_x += agent_fx * DELTA_TIME * PRED_SPEED_ADVANTAGE;
  agent_y += agent_fy * DELTA_TIME * PRED_SPEED_ADVANTAGE;


  //Bound the position within the environment 
  boundPosition(agent_x, agent_y, FLAMEGPU->environment.getProperty<float>("MIN_POSITION"), FLAMEGPU->environment.getProperty<float>("MAX_POSITION"));

  //Update the agents position and velocity
  FLAMEGPU->setVariable<float>("x",agent_x);
  FLAMEGPU->setVariable<float>("y",agent_y);
  FLAMEGPU->setVariable<float>("fx",agent_fx);
  FLAMEGPU->setVariable<float>("fy",agent_fy);

  //reduce life by one unit of energy
  FLAMEGPU->setVariable<int>("life",FLAMEGPU->getVariable<int>("life")-1);

  return ALIVE;
}
"""
"""
  predator_eat_or_starve agent function for Predator agents, which uses prey_eaten messages recieved with matching id to increase its own life
"""
predator_eat_or_starve = """
FLAMEGPU_AGENT_FUNCTION(predator_eat_or_starve, MsgSpatial2D, MsgNone) {
  int agent_life = FLAMEGPU->getVariable<int>("life");
  int agent_id = FLAMEGPU->getVariable<int>("id");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  int message_id = 0;
  const int GAIN_FROM_FOOD_PREDATOR = FLAMEGPU->environment.getProperty<int>("GAIN_FROM_FOOD_PREDATOR");

  // Iterate prey eaten messages, accumulating relevant data and counts.
  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y)) {
    message_id = message.getVariable<int>("predator_id");
    if (agent_id == message_id) {
      agent_life += GAIN_FROM_FOOD_PREDATOR;
    }
  }

  //if one or more predators were within killing distance then notify the nearest predator that it has eaten this predator via a predator eaten message.
  if (agent_life < 1) {
    return DEAD;
  }
  else {
   FLAMEGPU->setVariable<int>("life", agent_life);
  }

  //return eaten value to remove dead (eaten == 1) agents from the simulation
  return ALIVE;
}
"""
"""
  predator_reproduce agent function for Predator agents, which uses the reproduction chance environment variable and a random number to possibly generate a new predator agent in a random location. 
    The parent agent's life value is halved and shared with the offspring prey agent.
"""
predator_reproduce="""
FLAMEGPU_AGENT_FUNCTION(predator_reproduce, MsgNone, MsgNone) {
  float reproduction = FLAMEGPU->random.uniform<float>();
  const float PREDATOR_REPRODUCTION_CHANCE = FLAMEGPU->environment.getProperty<float>("PREDATOR_REPRODUCTION_CHANCE");
  if (reproduction<PREDATOR_REPRODUCTION_CHANCE) {
    const float BOUNDS_WIDTH = FLAMEGPU->environment.getProperty<float>("BOUNDS_WIDTH");
    const float SPEED_DIFFERENCE = FLAMEGPU->environment.getProperty<float>("SPEED_DIFFERENCE");
    int new_id = FLAMEGPU->getVariable<int>("id") * 3;
    FLAMEGPU->agent_out.setVariable<int>("id", new_id);
    reproduction = (FLAMEGPU->random.uniform<float>()*BOUNDS_WIDTH)-(BOUNDS_WIDTH/2.0);
    FLAMEGPU->agent_out.setVariable<float>("x", reproduction);
    reproduction = (FLAMEGPU->random.uniform<float>()*BOUNDS_WIDTH)-(BOUNDS_WIDTH/2.0);
    FLAMEGPU->agent_out.setVariable<float>("y", reproduction);
    reproduction = (FLAMEGPU->random.uniform<float>()*SPEED_DIFFERENCE)-(SPEED_DIFFERENCE/2.0);
    FLAMEGPU->agent_out.setVariable<float>("fx", reproduction);
    reproduction = (FLAMEGPU->random.uniform<float>()*SPEED_DIFFERENCE)-(SPEED_DIFFERENCE/2.0);
    FLAMEGPU->agent_out.setVariable<float>("fy", reproduction);
    FLAMEGPU->agent_out.setVariable<float>("type", FLAMEGPU->getVariable<float>("type"));
    FLAMEGPU->agent_out.setVariable<float>("steer_x", 0.0);
    FLAMEGPU->agent_out.setVariable<float>("steer_y", 0.0);
    int half_life = (int)(FLAMEGPU->getVariable<int>("life")/2.0);
    FLAMEGPU->agent_out.setVariable<int>("life", half_life);
    FLAMEGPU->setVariable<int>("life", half_life);
  }
  return ALIVE;
}
"""
"""
  GRASS
  grass_output_location_data agent function for Grass agents, which outputs publicly visible properties to a message list
"""
grass_output_location_data = """
FLAMEGPU_AGENT_FUNCTION(grass_output_location_data, MsgNone, MsgSpatial2D) {
  // Output each grass agent's location (and other visible properties if implemented)
  FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
  FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
  FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
  return ALIVE;
  }
"""
"""
  grass_eaten agent function for Grass agents, which checks if nearby prey are within eating distance and messages the closest to indicate a success. 
    The grass agent becomes unavailable to be eaten  for a number of simulation steps equal to the grass regrowth environment property
"""
grass_eaten = """
FLAMEGPU_HOST_DEVICE_FUNCTION float vec2Length(const float x, const float y) {
  return sqrtf(x * x + y * y);
}
FLAMEGPU_AGENT_FUNCTION(grass_eaten, MsgSpatial2D, MsgSpatial2D) {
  //if (FLAMEGPU->getVariable<int>("available")==1) {
  int eaten = 0;
  int prey_id = -1;
  float closest_prey = FLAMEGPU->environment.getProperty<float>("GRASS_EAT_DISTANCE");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float message_x = 0.0;
  float message_y = 0.0;
  int message_id = 0;
  float distance = 0.0;

  // Iterate location messages, accumulating relevant data and counts.
  for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y)) {
    message_x = message.getVariable<float>("x");
    message_y = message.getVariable<float>("y");
    message_id = message.getVariable<int>("id");
    distance = vec2Length(agent_x - message_x, agent_y - message_y);
    if (distance < closest_prey) {
      prey_id = message_id;
      closest_prey = distance;
      eaten = 1;
    }
  }

  //if one or more preys were within killing distance then notify the nearest prey that it has eaten this grass via a grass eaten message.
  if (eaten) {
    FLAMEGPU->message_out.setVariable<int>("prey_id", prey_id);
    FLAMEGPU->message_out.setVariable<float>("x", agent_x);
    FLAMEGPU->message_out.setVariable<float>("y", agent_y);


    //FLAMEGPU->setVariable<float>("type", 0.0);
    //FLAMEGPU->setVariable<int>("available", 0);
    FLAMEGPU->agent_out.setVariable<int>("id",FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->agent_out.setVariable<float>("x",agent_x);
    FLAMEGPU->agent_out.setVariable<float>("y",agent_y);
    FLAMEGPU->agent_out.setVariable<int>("dead_cycles",0);
    FLAMEGPU->agent_out.setVariable<float>("type",3.0);
    return DEAD;
  }
  return ALIVE;
}
//}
"""
"""
  grass_growth agent function for Grass agents, which checks if the grass has been unavailable for suitable steps to regrow. If not dead_cycles increments
"""
grass_growth = """
FLAMEGPU_AGENT_FUNCTION(grass_growth, MsgNone, MsgNone) {
  int agent_dead_cycles = FLAMEGPU->getVariable<int>("dead_cycles");
  //int agent_availability = FLAMEGPU->getVariable<int>("available");
  const int GRASS_REGROW_CYCLES = FLAMEGPU->environment.getProperty<int>("GRASS_REGROW_CYCLES");
  
  if (agent_dead_cycles>=GRASS_REGROW_CYCLES) {
    //FLAMEGPU->setVariable<float>("type", 2.0);
    //FLAMEGPU->setVariable<int>("available", 1);
    //FLAMEGPU->setVariable<int>("dead_cycles", 0);
    FLAMEGPU->agent_out.setVariable<int>("id",FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->agent_out.setVariable<float>("x",FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->agent_out.setVariable<float>("y",FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->agent_out.setVariable<int>("dead_cycles",0);
    FLAMEGPU->agent_out.setVariable<float>("type",2.0);
    return DEAD;
  }
  FLAMEGPU->setVariable<int>("dead_cycles", agent_dead_cycles+1);
  //if (agent_availability==0) {
  //  FLAMEGPU->setVariable<int>("dead_cycles", agent_dead_cycles+1);
  //}

  return ALIVE;
}
"""

model = pyflamegpu.ModelDescription("Prey_Predator_Grass");


"""
  GLOBALS
"""
env = model.Environment();
# Population size to generate, if no agents are loaded from disk
env.newPropertyUInt("PREY_POPULATION_TO_GENERATE", PREY_POPULATION_SIZE);
env.newPropertyUInt("PREDATOR_POPULATION_TO_GENERATE", PREDATOR_POPULATION_SIZE);
env.newPropertyUInt("GRASS_POPULATION_TO_GENERATE", GRASS_POPULATION_SIZE);
env.newPropertyUInt("CURRENT_ID", 0);

# Number of steps to simulate
env.newPropertyUInt("STEPS", STEPS);

# Environment Bounds
env.newPropertyFloat("MIN_POSITION", -1.0);
env.newPropertyFloat("MAX_POSITION", +1.0);
env.newPropertyFloat("BOUNDS_WIDTH", 2.0);

# Interaction Radii
env.newPropertyFloat("PI", 3.1415);
env.newPropertyFloat("PRED_PREY_INTERACTION_RADIUS", 0.1);
env.newPropertyFloat("PREY_GROUP_COHESION_RADIUS", 0.2);
env.newPropertyFloat("SAME_SPECIES_AVOIDANCE_RADIUS", 0.035);
env.newPropertyFloat("GRASS_EAT_DISTANCE", 0.02);
env.newPropertyFloat("PREDATOR_KILL_DISTANCE", 0.02);

# Other globals
env.newPropertyFloat("DELTA_TIME", 0.001);
env.newPropertyFloat("PRED_SPEED_ADVANTAGE", 2.0);
env.newPropertyFloat("MIN_SPEED", -1.0);
env.newPropertyFloat("MAX_SPEED", +1.0);
env.newPropertyFloat("SPEED_DIFFERENCE", 2.0);

#Parameter globals
env.newPropertyFloat("PREY_REPRODUCTION_CHANCE", 0.05);
env.newPropertyFloat("PREDATOR_REPRODUCTION_CHANCE", 0.03);
env.newPropertyUInt("GAIN_FROM_FOOD_PREDATOR", 75);
env.newPropertyUInt("GAIN_FROM_FOOD_PREY", 50);
env.newPropertyUInt("GRASS_REGROW_CYCLES", 100);

"""
  Location messages
"""
grass_location_message = model.newMessageSpatial2D("grass_location_message");
# Set the range and bounds.
grass_location_message.setRadius(env.getPropertyFloat("PRED_PREY_INTERACTION_RADIUS"));
grass_location_message.setMin(env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"));
grass_location_message.setMax(env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"));
# A message to hold the location of an agent.
grass_location_message.newVariableInt("id");

prey_location_message = model.newMessageSpatial2D("prey_location_message");
# Set the range and bounds.
prey_location_message.setRadius(env.getPropertyFloat("PRED_PREY_INTERACTION_RADIUS"));
prey_location_message.setMin(env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"));
prey_location_message.setMax(env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"));
# A message to hold the location of an agent.
prey_location_message.newVariableInt("id");

predator_location_message = model.newMessageSpatial2D("predator_location_message");
# Set the range and bounds.
predator_location_message.setRadius(env.getPropertyFloat("PRED_PREY_INTERACTION_RADIUS"));
predator_location_message.setMin(env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"));
predator_location_message.setMax(env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"));
# A message to hold the location of an agent.
predator_location_message.newVariableInt("id");

"""
  Eaten messages
"""
prey_eaten_message = model.newMessageSpatial2D("prey_eaten_message");
# Set the range and bounds.
prey_eaten_message.setRadius(env.getPropertyFloat("PRED_PREY_INTERACTION_RADIUS"));
prey_eaten_message.setMin(env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"));
prey_eaten_message.setMax(env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"));
# A message to hold the eaten of an agent.
prey_eaten_message.newVariableInt("predator_id");

grass_eaten_message = model.newMessageSpatial2D("grass_eaten_message");
# Set the range and bounds.
grass_eaten_message.setRadius(env.getPropertyFloat("PRED_PREY_INTERACTION_RADIUS"));
grass_eaten_message.setMin(env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"));
grass_eaten_message.setMax(env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"));
# A message to hold the eaten of an agent.
grass_eaten_message.newVariableInt("prey_id");
    
"""
  Prey agent
"""
prey_agent = model.newAgent("Prey");
prey_agent.newVariableInt("id");
prey_agent.newVariableFloat("x");
prey_agent.newVariableFloat("y");
prey_agent.newVariableFloat("fx");
prey_agent.newVariableFloat("fy");
prey_agent.newVariableFloat("type");
prey_agent.newVariableFloat("steer_x");
prey_agent.newVariableFloat("steer_y");
prey_agent.newVariableInt("life");
prey_agent.newRTCFunction("prey_output_location_data", prey_output_location_data).setMessageOutput("prey_location_message");
prey_agent.newRTCFunction("prey_avoid_predators", prey_avoid_predators).setMessageInput("predator_location_message");
prey_agent.newRTCFunction("prey_flock", prey_flock).setMessageInput("prey_location_message");
prey_agent.newRTCFunction("prey_move", prey_move);
prey_agent.newRTCFunction("prey_eaten", prey_eaten).setMessageInput("predator_location_message");
prey_agent.Function("prey_eaten").setMessageOutput("prey_eaten_message");
prey_agent.Function("prey_eaten").setMessageOutputOptional(True);
prey_agent.Function("prey_eaten").setAllowAgentDeath(True)
prey_agent.newRTCFunction("prey_eat_or_starve", prey_eat_or_starve).setMessageInput("grass_eaten_message");
prey_agent.Function("prey_eat_or_starve").setAllowAgentDeath(True)
prey_agent.newRTCFunction("prey_reproduce", prey_reproduce).setAgentOutput(prey_agent);

"""
  Predator agent
"""
predator_agent = model.newAgent("Predator");
predator_agent.newVariableInt("id");
predator_agent.newVariableFloat("x");
predator_agent.newVariableFloat("y");
predator_agent.newVariableFloat("fx");
predator_agent.newVariableFloat("fy");
predator_agent.newVariableFloat("type");
predator_agent.newVariableFloat("steer_x");
predator_agent.newVariableFloat("steer_y");
predator_agent.newVariableInt("life");
predator_agent.newRTCFunction("predator_output_location_data", predator_output_location_data).setMessageOutput("predator_location_message");
predator_agent.newRTCFunction("predator_follow_prey", predator_follow_prey).setMessageInput("prey_location_message");
predator_agent.newRTCFunction("predator_avoidance", predator_avoidance).setMessageInput("predator_location_message");
predator_agent.newRTCFunction("predator_move", predator_move);
predator_agent.newRTCFunction("predator_eat_or_starve", predator_eat_or_starve).setMessageInput("prey_eaten_message");
predator_agent.Function("predator_eat_or_starve").setAllowAgentDeath(True);
predator_agent.newRTCFunction("predator_reproduce", predator_reproduce).setAgentOutput(predator_agent);

"""
  Grass agent
"""
grass_agent = model.newAgent("Grass");
uagrass_agent = model.newAgent("Unavailable_Grass");
grass_agent.newVariableInt("id");
grass_agent.newVariableFloat("x");
grass_agent.newVariableFloat("y");
grass_agent.newVariableFloat("type");
grass_agent.newVariableInt("dead_cycles");
grass_agent.newRTCFunction("grass_output_location_data", grass_output_location_data).setMessageOutput("grass_location_message");
grass_agent.newRTCFunction("grass_eaten", grass_eaten).setMessageInput("prey_location_message");
grass_agent.Function("grass_eaten").setMessageOutput("grass_eaten_message");
grass_agent.Function("grass_eaten").setMessageOutputOptional(True);
grass_agent.Function("grass_eaten").setAllowAgentDeath(True);
grass_agent.Function("grass_eaten").setAgentOutput(uagrass_agent);
#grass_agent.newRTCFunction("grass_growth", grass_growth);

"""
  Unavailable Grass agent
"""

uagrass_agent.newVariableInt("id");
uagrass_agent.newVariableFloat("x");
uagrass_agent.newVariableFloat("y");
uagrass_agent.newVariableFloat("type");
uagrass_agent.newVariableInt("dead_cycles");
uagrass_agent.newRTCFunction("grass_growth", grass_growth).setAllowAgentDeath(True);
uagrass_agent.Function("grass_growth").setAgentOutput(grass_agent);


"""
  Population initialisation functions
"""
# Add init function for creating prey population with random initial location and life within large range for testing purposes
class initPreyPopulation(pyflamegpu.HostFunctionCallback):
  def run(self,FLAMEGPU):
    populationSize = FLAMEGPU.environment.getPropertyUInt("PREY_POPULATION_TO_GENERATE");
    min_pos = FLAMEGPU.environment.getPropertyFloat("MIN_POSITION");
    max_pos = FLAMEGPU.environment.getPropertyFloat("MAX_POSITION");
    min_speed = FLAMEGPU.environment.getPropertyFloat("MIN_SPEED");
    max_speed = FLAMEGPU.environment.getPropertyFloat("MAX_SPEED");
    current_id = FLAMEGPU.environment.getPropertyUInt("CURRENT_ID");
    for i in range(populationSize):
      instance = FLAMEGPU.agent("Prey").newAgent();
      instance.setVariableInt("id", current_id+i);
      instance.setVariableFloat("x", random.uniform(min_pos, max_pos));
      instance.setVariableFloat("y", random.uniform(min_pos, max_pos));
      instance.setVariableFloat("type", 2.0);
      instance.setVariableFloat("fx", random.uniform(min_speed, max_speed));
      instance.setVariableFloat("fy", random.uniform(min_speed, max_speed));
      instance.setVariableFloat("steer_x", 0.0);
      instance.setVariableFloat("steer_y", 0.0);
      instance.setVariableInt("life", random.randint(200,5000));
    FLAMEGPU.environment.setPropertyUInt("CURRENT_ID", current_id+i)
    return
# Add init function for creating predator population with random initial location and life within large range for testing purposes
class initPredatorPopulation(pyflamegpu.HostFunctionCallback):
  def run(self,FLAMEGPU):
    populationSize = FLAMEGPU.environment.getPropertyUInt("PREDATOR_POPULATION_TO_GENERATE");
    min_pos = FLAMEGPU.environment.getPropertyFloat("MIN_POSITION");
    max_pos = FLAMEGPU.environment.getPropertyFloat("MAX_POSITION");
    min_speed = FLAMEGPU.environment.getPropertyFloat("MIN_SPEED");
    max_speed = FLAMEGPU.environment.getPropertyFloat("MAX_SPEED");
    current_id = FLAMEGPU.environment.getPropertyUInt("CURRENT_ID");
    for i in range(populationSize):
      instance = FLAMEGPU.agent("Predator").newAgent();
      instance.setVariableInt("id", current_id+i);
      instance.setVariableFloat("x", random.uniform(min_pos, max_pos));
      instance.setVariableFloat("y", random.uniform(min_pos, max_pos));
      instance.setVariableFloat("type", 0.0);
      instance.setVariableFloat("fx", random.uniform(min_speed, max_speed));
      instance.setVariableFloat("fy", random.uniform(min_speed, max_speed));
      instance.setVariableFloat("steer_x", 0.0);
      instance.setVariableFloat("steer_y", 0.0);
      instance.setVariableInt("life", random.randint(200,5000));
    FLAMEGPU.environment.setPropertyUInt("CURRENT_ID", current_id+i)
    return
# Add init function for creating grass population with random initial location
class initGrassPopulation(pyflamegpu.HostFunctionCallback):
  def run(self,FLAMEGPU):
    populationSize = FLAMEGPU.environment.getPropertyUInt("GRASS_POPULATION_TO_GENERATE");
    min_pos = FLAMEGPU.environment.getPropertyFloat("MIN_POSITION");
    max_pos = FLAMEGPU.environment.getPropertyFloat("MAX_POSITION");
    current_id = FLAMEGPU.environment.getPropertyUInt("CURRENT_ID");
    for i in range(populationSize):
      instance = FLAMEGPU.agent("Grass").newAgent();
      instance.setVariableInt("id", current_id+i);
      instance.setVariableFloat("x", random.uniform(min_pos, max_pos));
      instance.setVariableFloat("y", random.uniform(min_pos, max_pos));
      instance.setVariableFloat("type", 1.0);
      instance.setVariableInt("dead_cycles", 0);
      #instance.setVariableInt("available", 1);
    FLAMEGPU.environment.setPropertyUInt("CURRENT_ID", current_id+i)
    return

# Add function callback to INIT functions for population generation
initialPreyPopulation = initPreyPopulation();
model.addInitFunctionCallback(initialPreyPopulation);
initialPredatorPopulation = initPredatorPopulation();
model.addInitFunctionCallback(initialPredatorPopulation);
initialGrassPopulation = initGrassPopulation();
model.addInitFunctionCallback(initialGrassPopulation);

"""
  Example step function for later reference
"""
# externalCounter = 0
# class IncrementCounter(pyflamegpu.HostFunctionCallback):
#     """
#     pyflamegpu requires step functions to be a class which extends the StepFunction base class.
#     This class must extend the handle function
#     """

#     # Define Python class 'constructor'
#     def __init__(self):
#         super().__init__()

#     # Override C++ method: virtual void run(FLAMEGPU_HOST_API*);
#     def run(self, host_api):
#         global externalCounter
#         print ("Hello from step function on step:",externalCounter)
#         externalCounter += 1
# inc = IncrementCounter()
# model.addStepFunctionCallback(inc)

"""
  Control flow
"""    
# Layer #1
model.newLayer("L1").addAgentFunction("Prey", "prey_output_location_data");
model.Layer("L1").addAgentFunction("Predator", "predator_output_location_data");
model.Layer("L1").addAgentFunction("Grass", "grass_output_location_data");
# Layer #2
model.newLayer("L2").addAgentFunction("Predator", "predator_follow_prey");
model.Layer("L2").addAgentFunction("Prey", "prey_avoid_predators");
# Layer #3
model.newLayer("L3").addAgentFunction("Prey", "prey_flock");
model.Layer("L3").addAgentFunction("Predator", "predator_avoidance");
# Layer #4
model.newLayer("L4").addAgentFunction("Prey", "prey_move");
model.Layer("L4").addAgentFunction("Predator", "predator_move");
# Layer #5
model.newLayer("L5").addAgentFunction("Grass", "grass_eaten");
model.Layer("L5").addAgentFunction("Prey", "prey_eaten");
# Layer #6
model.newLayer("L6").addAgentFunction("Prey", "prey_eat_or_starve");
model.Layer("L6").addAgentFunction("Predator", "predator_eat_or_starve");
# Layer #7
model.newLayer("L7").addAgentFunction("Unavailable_Grass", "grass_growth");
model.Layer("L7").addAgentFunction("Prey", "prey_reproduce");
model.Layer("L7").addAgentFunction("Predator", "predator_reproduce");

# Create and configure logging details 
logging_config = pyflamegpu.LoggingConfig(model);
logging_config.logEnvironment("PREDATOR_KILL_DISTANCE");
logging_config.logEnvironment("MIN_SPEED");
logging_config.logEnvironment("GRASS_EAT_DISTANCE");
logging_config.logEnvironment("CURRENT_ID");
prey_agent_log = logging_config.agent("Prey");
prey_agent_log.logCount();
prey_agent_log.logMeanFloat("x");
prey_agent_log.logMeanFloat("y");
prey_agent_log.logMeanFloat("fx");
prey_agent_log.logMeanFloat("fy");
prey_agent_log.logMeanInt("life");
prey_agent_log.logStandardDevInt("life");
predator_agent_log = logging_config.agent("Predator");
predator_agent_log.logCount();
predator_agent_log.logMeanFloat("x");
predator_agent_log.logMeanFloat("y");
predator_agent_log.logMeanFloat("fx");
predator_agent_log.logMeanFloat("fy");
predator_agent_log.logMeanInt("life");
predator_agent_log.logStandardDevInt("life");
grass_agent_log = logging_config.agent("Grass");
grass_agent_log.logCount();
#grass_agent_log.logMeanInt("available");
grass_agent_log.logMeanInt("dead_cycles");
grass_agent_log.logStandardDevInt("dead_cycles");
uagrass_agent_log = logging_config.agent("Unavailable_Grass");
uagrass_agent_log.logCount();
#grass_agent_log.logMeanInt("available");
uagrass_agent_log.logMeanInt("dead_cycles");
uagrass_agent_log.logStandardDevInt("dead_cycles");
step_log = pyflamegpu.StepLoggingConfig(logging_config);
step_log.setFrequency(1);


"""
  Create Model Runner
"""  
if ENSEMBLE: 
  """
  Create Run Plan Vector
  """   
  run_plan_vector = pyflamegpu.RunPlanVec(model, ENSEMBLE_RUNS);
  run_plan_vector.setSteps(env.getPropertyUInt("STEPS"));
  simulation_seed = random.randint(0,99999);
  run_plan_vector.setRandomSimulationSeed(simulation_seed,1000);
  simulation = pyflamegpu.CUDAEnsemble(model);
else:
  simulation = pyflamegpu.CUDASimulation(model);
  if not VISUALISATION:
    simulation.SimulationConfig().steps = STEPS;

simulation.setStepLog(step_log);
simulation.setExitLog(logging_config)

"""
  Create Visualisation
"""
if pyflamegpu.VISUALISATION and VISUALISATION and not ENSEMBLE:
    visualisation = simulation.getVisualisation();
    # Configure vis
    envWidth = env.getPropertyFloat("MAX_POSITION") - env.getPropertyFloat("MIN_POSITION");
    INIT_CAM = env.getPropertyFloat("MAX_POSITION") * 1.5;
    visualisation.setInitialCameraLocation(0.0, 0.0, INIT_CAM);
    visualisation.setCameraSpeed(0.002 * envWidth);
    circ_prey_agt = visualisation.addAgent("Prey");
    circ_predator_agt = visualisation.addAgent("Predator");
    circ_grass_agt = visualisation.addAgent("Grass");
    circ_uagrass_agt = visualisation.addAgent("Unavailable_Grass");
    # Position vars are named x, y, z; so they are used by default
    circ_prey_agt.setModel(pyflamegpu.ICOSPHERE);
    circ_predator_agt.setModel(pyflamegpu.ICOSPHERE);
    circ_grass_agt.setModel(pyflamegpu.ICOSPHERE);
    circ_uagrass_agt.setModel(pyflamegpu.ICOSPHERE);
    circ_prey_agt.setModelScale(env.getPropertyFloat("PRED_PREY_INTERACTION_RADIUS")/7.5);
    circ_prey_agt.setColor(pyflamegpu.BLUE);
    circ_predator_agt.setModelScale(env.getPropertyFloat("PRED_PREY_INTERACTION_RADIUS")/7.5);
    circ_predator_agt.setColor(pyflamegpu.RED);
    circ_grass_agt.setModelScale(env.getPropertyFloat("PRED_PREY_INTERACTION_RADIUS")/7.5);
    circ_grass_agt.setColor(pyflamegpu.GREEN);
    circ_uagrass_agt.setModelScale(env.getPropertyFloat("PRED_PREY_INTERACTION_RADIUS")/7.5);
    circ_uagrass_agt.setColor(pyflamegpu.WHITE);
    visualisation.activate();

"""
  Execution
"""
if ENSEMBLE:
    simulation.simulate(run_plan_vector);
else:
    simulation.simulate();

"""
  Export Pop
"""
# simulation.exportData("end.xml");

# Join Visualisation
if pyflamegpu.VISUALISATION and VISUALISATION and not ENSEMBLE:
    visualisation.join();



# Deal with logs
if ENSEMBLE:
    logs = simulation.getLogs();
else:
    logs = simulation.getRunLog();


if ENSEMBLE:
    agent_counts = [None]*ENSEMBLE_RUNS
    prey_life_mean = [None]*ENSEMBLE_RUNS
    predator_life_mean = [None]*ENSEMBLE_RUNS
    prey_life_std = [None]*ENSEMBLE_RUNS
    predator_life_std = [None]*ENSEMBLE_RUNS
    grass_available_mean = [None]*ENSEMBLE_RUNS
    grass_dead_cycles_mean = [None]*ENSEMBLE_RUNS
    grass_dead_cycles_std = [None]*ENSEMBLE_RUNS
    pred_kill = [None]*ENSEMBLE_RUNS
    min_speed = [None]*ENSEMBLE_RUNS
    grass_eat = [None]*ENSEMBLE_RUNS
    current_id = [None]*ENSEMBLE_RUNS
    # Read logs
    for i in range(len(logs)):
      sl = logs[i].getStepLog();
      agent_counts[i] = [["ensemble_run_"+str(i)+"_prey"],["ensemble_run_"+str(i)+"_pred"],["ensemble_run_"+str(i)+"_gras"]];
      prey_life_mean[i] = ["ensemble_run_"+str(i)];
      prey_life_std[i] = ["ensemble_run_"+str(i)];
      predator_life_mean[i] = ["ensemble_run_"+str(i)];
      predator_life_std[i] = ["ensemble_run_"+str(i)];
      grass_available_mean[i] = ["ensemble_run_"+str(i)];
      grass_dead_cycles_mean[i] = ["ensemble_run_"+str(i)];
      grass_dead_cycles_std[i] = ["ensemble_run_"+str(i)];
      pred_kill[i] = ["ensemble_run_"+str(i)];
      min_speed[i] = ["ensemble_run_"+str(i)];
      grass_eat[i] = ["ensemble_run_"+str(i)];
      current_id[i] = ["ensemble_run_"+str(i)];
      for step in sl:
        # Collect step data
        prey_agents = step.getAgent("Prey");
        predator_agents = step.getAgent("Predator");
        grass_agents = step.getAgent("Grass");
        pred_kill[i].append(step.getEnvironmentPropertyFloat("PREDATOR_KILL_DISTANCE"));
        min_speed[i].append(step.getEnvironmentPropertyFloat("MIN_SPEED"));
        grass_eat[i].append(step.getEnvironmentPropertyFloat("GRASS_EAT_DISTANCE"));
        current_id[i].append(step.getEnvironmentPropertyUInt("CURRENT_ID"));
        # Collect agent data from step
        prey_life_mean[i].append(prey_agents.getMean("life"));
        prey_life_std[i].append(prey_agents.getStandardDev("life"));
        agent_counts[i][0].append(prey_agents.getCount());

        predator_life_mean[i].append(predator_agents.getMean("life"));
        predator_life_std[i].append(predator_agents.getStandardDev("life"));
        agent_counts[i][1].append(predator_agents.getCount());

        grass_dead_cycles_mean[i].append(grass_agents.getMean("dead_cycles"));
        grass_dead_cycles_std[i].append(grass_agents.getStandardDev("dead_cycles"));
        #agent_counts[i][2].append(int(grass_agents.getMean("available")*grass_agents.getCount()));
        agent_counts[i][2].append(grass_agents.getCount());
        #grass_available_mean[i].append(grass_agents.getMean("available"));
        #agent_counts[i][2].append(grass_agents.getCount());

    # Print log data    
    # print("prey life mean",prey_life_mean);
    # print("prey life std",prey_life_std);
    # print()
    # print("predator life mean",predator_life_mean);
    # print("predator life std",predator_life_std);
    # print()
    # print("grass available mean",grass_available_mean);
    # print("grass dead cycles mean",grass_dead_cycles_mean);
    # print("grass dead cycles std",grass_dead_cycles_std);
    print()


    #Print warning data
    print("Agent counts per step per ensemble run")
    for j in range(len(agent_counts)):
      for k in range(len(agent_counts[j])):
        print(agent_counts[j][k])
    print()
    print("Predator kill distance environment variable (cut to 9 steps/40 for readability)")
    for l in range(len(pred_kill)):
      print(pred_kill[l][:10])
    print()
    print("Min speed environment variable (cut to 9 steps/40 for readability)")
    for m in range(len(min_speed)):
      print(min_speed[m][:10])
    print()
    print("Grass eat distance environment variable (cut to 9 steps/40 for readability) - No curve warning")
    for n in range(len(grass_eat)):
      print(grass_eat[n][:10])
    #print("grass eat distance environment property",grass_eat);
    #print("current_id environment property",current_id);


    """
      Boid graph generation for future reference
    """
    # Generate graphs 
    # for j in range(ENSEMBLE_RUNS):
    #     # Plot 3d graph of average flock position over simulation for individual model run
    #     fig = plt.figure(figsize=(8,8));
    #     ax = fig.gca(projection='3d');
    #     ax.set_xlabel("Model environment x");
    #     ax.set_ylabel("Model environment y");
    #     ax.set_zlabel("Model environment z");
    #     fig.suptitle("Ensemble run "+str(j)+" boids mean flock positions",fontsize=16);
    #     label = "Boids mean flock position, ensemble run "+str(j);
    #     fname = "average_flock_positions_run"+str(j)+".png";
    #     # Position start and finish flock position text
    #     for k in text_pos[j]:
    #         ax.text(k[0],k[1],k[2],k[3],None);
    #     ax.plot(positions_mean[j][0], positions_mean[j][1], positions_mean[j][2], label=label);
    #     ax.set_xlim3d([-1.0,1.0]);
    #     ax.set_ylim3d([-1.0,1.0]);
    #     ax.set_zlim3d([-1.0,1.0]);
    #     ax.legend();
    #     plt.savefig(fname,format='png');
    #     plt.close(fig);

    #     # Plot graphs for average of each fx, fy, and fz with standard deviation error bars
    #     steplist = range(STEPS);
    #     fig,(axx,axy,axz) = plt.subplots(1,3, figsize=(21,6));
    #     fig.suptitle("Ensemble run "+str(j)+" mean boid velocities with std errorbar",fontsize=16);
    #     velfname = "mean_velocities_run"+str(j)+".png";
    #     axx.errorbar(steplist,velocities_mean[j][0],yerr=velocities_std[j][0],elinewidth=0.5,capsize=1.0);
    #     axx.set_xlabel("Simulation step");
    #     axx.set_ylabel("Boid agents average fx");
    #     axy.errorbar(steplist,velocities_mean[j][1],yerr=velocities_std[j][1],elinewidth=0.5,capsize=1.0);
    #     axy.set_xlabel("Simulation step");
    #     axy.set_ylabel("Boid agents average fy");
    #     axz.errorbar(steplist,velocities_mean[j][2],yerr=velocities_std[j][2],elinewidth=0.5,capsize=1.0);
    #     axz.set_xlabel("Simulation step");
    #     axz.set_ylabel("Boid agents average fz");
    #     plt.savefig(velfname,format='png');
    #     plt.close(fig);

    # # Plot every model in esemble's average flock position over simulation on the same 3d graph
    # fig = plt.figure(figsize=(12,12));
    # fig.suptitle("Ensemble Boids mean flock positions",fontsize=16);
    # ax = fig.gca(projection='3d');
    # ax.set_xlabel("Model environment x");
    # ax.set_ylabel("Model environment y");
    # ax.set_zlabel("Model environment z");
    # fname = "ensemble_average_flock_positions.png";
    # ## Plot start and finish text for each flock path ---VERY CLUTTERED---
    # # for i in text_pos:
    # #     for k in i:
    # #         ax.text(k[0],k[1],k[2],k[3],'x');
    # jcount = 0;
    # for j in positions_mean:
    #     label1 = "Run "+str(jcount);
    #     ax.plot(j[0], j[1], j[2], label=label1);
    #     jcount+=1;
    # #ax.set_xlim3d([-1.0,1.0]);
    # #ax.set_ylim3d([-1.0,1.0]);
    # #ax.set_zlim3d([-1.0,1.0]);
    # ax.legend();
    # plt.savefig(fname,format='png');
    # plt.close(fig);
else:
    steps = logs.getStepLog();
    prey_agent_counts = [None]*len(steps)
    predator_agent_counts = [None]*len(steps)
    grass_agent_counts = [None]*len(steps)
    counter = 0;
    for step in steps:
        stepcount = step.getStepCount();
        prey_agents = step.getAgent("Prey");
        predator_agents = step.getAgent("Predator");
        grass_agents = step.getAgent("Grass");
        prey_agent_counts[counter] = prey_agents.getCount();
        predator_agent_counts[counter] = predator_agents.getCount();
        grass_agent_counts[counter] = grass_agents.getCount();
        counter+=1;
    print()
    print("Agent counts per step")
    for j in range(len(steps)):
      print("step",j,"prey",prey_agent_counts[j],"predators",predator_agent_counts[j],"grass",grass_agent_counts[j]) 