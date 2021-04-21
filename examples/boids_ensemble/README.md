# FLAMEGPU2 Reproducible Search Experimentation
This repository acts as an example to be used as a template for creating standalone FLAMEGPU2 projects runnning parameter search ensemble experiments of python(SWIG) based models.

[FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2_dev) is downloaded via CMake and configured as a dependency of the project.

Currently, it uses the version of FLAMEGPU2 from master, this can be changed locally by setting the CMake variable `FLAMEGPU2_Version` to point to a different git branch or tag. You can also change it for all users, by changing `cmake/flamegpu2.cmake:5` which provides the default value.

### Example Python model

An ensemble of boids models can be found at `/examples/boids_ensemble`. Running this script will execute a single visual simulation of the model.

### Running model ensembles

Within the script variables: `ENSEMBLE_RUNS`, `POPULATION_SIZE` and `STEPS` may be edited to alter: the number of models simulated; boid agent population size; and how many simulation steps each model in to be run for respectively.

To perform an ensemble experiment a batch file `run_boids_ensemble.bat` has been provided. Alternatively a call to `boids_ensemble.py` within a python environment should be made.

### Setting up a search experiment

Commented out in  `/examples/boids_ensemble` is an example of how to set up a simple search experiment for the boids model. 

The experiment generator provides an interface for setting up an experiment for a particular model, which will then automatically generate and run an ensemble of the desired model with inital parameter values as specified during the experiment setup.

This requires user specification of a valid way to generate intial states for the model. Global variables and all variables for each population of agents, as well as the initial population sizes for these agents, must be specified.

The classes `AgentPopulation()` and `InitialStateGenerator()` are intended to be used for this purpose.

After designing the python(SWIG) model agent populations should be provided for each population of agents, including agents of the same type but initialised in a different state. For example:

```
population1 = experiment_generator.AgentPopulation();
population1.setDefaultState('Default1');
population1.setPopSizeRandom((256,512));
```

### Setting up a genetic algorithm

### Dependencies

The dependencies below are required for building FLAME GPU 2.

Only documentation can be built without the required dependencies (however Doxygen is still required).

### Required

* [CMake](https://cmake.org/) >= 3.12
  * CMake 3.16 is known to have issues on certain platforms
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) >= 9.0
* [git](https://git-scm.com/): Required by CMake for downloading dependencies
* *Linux:*
  * [make](https://www.gnu.org/software/make/)
  * gcc/g++ >= 6 (version requirements [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements))
      * gcc/g++ >= 7 required for the test suite 
* *Windows:*
  * Visual Studio 2015 or higher (2019 preferred)


## Building FLAME GPU 2

FLAME GPU 2 uses [CMake](https://cmake.org/), as a cross-platform process, for configuring and generating build directives, e.g. `Makefile` or `.vcxproj`. This is used to build the FLAMEGPU2 library, examples, tests and documentation.

Below the core commands are provided, for the full guide refer to the main [FLAMEGPU2 guide](https://github.com/FLAMEGPU/FLAMEGPU2_dev/blob/master/README.md).

When building it is required to enable python swig in order to run python based models.

### Linux

Under Linux, `cmake` can be used to generate makefiles specific to your system:

```
mkdir -p build && cd build
cmake .. 
make -j8
```

The option `-j8` enables parallel compilation using upto 8 threads, this is recommended to improve build times.

By default a `Makefile` for the `Release` build configuration will be generated.

Alternatively, using `-DCMAKE_BUILD_TYPE=`, `Debug` or `Profile` build configurations can be generated:
 
```
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Profile
make -j8
```

### Windows

*Note: If installing CMake on Windows ensure CMake is added to the system path, allowing `cmake` to be used via `cmd`, this option is disabled within the installer by default.*

When generating Visual studio project files, using `cmake` (or `cmake-gui`), the platform **must** be specified as `x64`.

Using `cmake` this takes the form `-A x64`:

```
mkdir build && cd build
cmake .. -A x64
ALL_BUILD.sln
```