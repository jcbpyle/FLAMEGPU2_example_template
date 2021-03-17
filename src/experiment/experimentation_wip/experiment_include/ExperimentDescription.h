#ifndef INCLUDE_FLAMEGPU_EXPERIMENT_EXPERIMENTDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_EXPERIMENT_EXPERIMENTDESCRIPTION_H_

#include <unordered_map>
#include <list>
#include <memory>
#include <typeindex>
#include <set>
#include <string>

#include "flamegpu/experiment/ExperimentData.h"


struct ExperimentData;

/**
 * This class represents the description of a reproducible parameter search experiment and optimisation via genetic algorithm for a FLAMEGPU model
 * This is the initial class that should be created by a modeller
 * @see ExperimentData The internal data store for this class
 */
class ExperimentDescription {
    
 public:
    /**
     * Constructor
     * @param experiment_name Name of the experiment, this must be unique between experiments
     */
    explicit ExperimentDescription(const std::string &experiment_name);
    /**
     * Default copy constructor, not implemented
     */
    ExperimentDescription(const ExperimentDescription &other_experiment) = delete;
    /**
     * Default move constructor, not implemented
     */
    ExperimentDescription(ExperimentDescription &&other_experiment) noexcept = delete;
    /**
     * Default copy assignment, not implemented
     */
    ExperimentDescription& operator=(const ExperimentDescription &other_experiment) = delete;
    /**
     * Default move assignment, not implemented
     */
    ExperimentDescription& operator=(ExperimentDescription &&other_experiment) noexcept = delete;
    /**
     * Equality operator, checks whether ExperimentDescription hierarchies are functionally the same
     * @returns True when experiments are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const ExperimentDescription& rhs) const;
    /**
     * Equality operator, checks whether ExperimentDescription hierarchies are functionally different
     * @returns True when experiments are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const ExperimentDescription& rhs) const;

#ifdef SWIG
    /**
     * Adds an init function callback to the simulation. The callback objects is similar to adding via addInitFunction
     * however the runnable function is encapsulated within an object which permits cross language support in swig.
     * Init functions execute once before the simulation begins
     * @param func_callback Pointer to the desired init function callback
     * @throws InvalidHostFunc If the init function has already been added to this model description
     * @note There is no guarantee on the order in which multiple init functions will be executed
     */
    //inline void addInitFunctionCallback(HostFunctionCallback *func_callback);
#endif

    /**
     * @return The experiment's name
     */
    std::string getName() const;

    private:
        std::shared_ptr<ExperimentData> experiment_pointer;
};

#ifdef SWIG
// void ExperimentDescription::addInitFunctionCallback(HostFunctionCallback* func_callback) {
//     if (!model->initFunctionCallbacks.insert(func_callback).second) {
//             THROW InvalidHostFunc("Attempted to add same init function callback twice,"
//                 "in ModelDescription::addInitFunctionCallback()");
//         }
// }
// void ExperimentDescription::addStepFunctionCallback(HostFunctionCallback* func_callback) {
//     if (!model->stepFunctionCallbacks.insert(func_callback).second) {
//             THROW InvalidHostFunc("Attempted to add same step function callback twice,"
//                 "in ModelDescription::addStepFunctionCallback()");
//         }
// }
// void ExperimentDescription::addExitFunctionCallback(HostFunctionCallback* func_callback) {
//     if (!model->exitFunctionCallbacks.insert(func_callback).second) {
//             THROW InvalidHostFunc("Attempted to add same exit function callback twice,"
//                 "in ModelDescription::addExitFunctionCallback()");
//         }
// }
// void ExperimentDescription::addExitConditionCallback(HostFunctionConditionCallback *func_callback) {
//     if (!model->exitConditionCallbacks.insert(func_callback).second) {
//             THROW InvalidHostFunc("Attempted to add same exit condition callback twice,"
//                 "in ModelDescription::addExitConditionCallback()");
//         }
// }
#endif

#endif  // INCLUDE_FLAMEGPU_EXPERIMENT_EXPERIMENTDESCRIPTION_H_
