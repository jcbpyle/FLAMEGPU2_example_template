#ifndef INCLUDE_EXPERIMENTATION_EXPERIMENTDATA_H_
#define INCLUDE_EXPERIMENTATION_EXPERIMENTDATA_H_

#include <unordered_map>
#include <list>
#include <memory>
#include <typeindex>
#include <set>
#include <string>

#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/runtime/flamegpu_host_api_macros.h"
#include "flamegpu/runtime/messaging/BruteForce.h"


class HostFunctionCallback;
class HostFunctionConditionCallback;
class EnvironmentDescription;
struct AgentData;
struct LayerData;
struct SubModelData;

/**
 * This is the internal data store for ModelDescription
 * Users should only access that data stored within via an instance of ModelDescription
 */
struct ExperimentData : std::enable_shared_from_this<ExperimentData>{
    virtual ~ExperimentData();
    /**
     * Default state, all agents and agent functions begin in/with this state
     */
    static const char *DEFAULT_STATE;  // "default"
    /**
     * Description needs full access
     */
    friend class ExperimentDescription;
    /**
     * Common size type used in the definition of models
     */
    typedef unsigned int size_type;
    
    
    /**
     * The name of the model
     * This must be unique among Simulation (e.g. CUDASimulation) instances
     */
    std::string name;
    
    /**
     * Equality operator, checks whether ModelData hierarchies are functionally the same
     * @returns True when models are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const ExperimentData& rhs) const;
    /**
     * Equality operator, checks whether ModelData hierarchies are functionally different
     * @returns True when models are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const ExperimentData& rhs) const;


 protected:
    /**
     * Copy constructor
     * This should only be called via clone();
     */
    explicit ExperimentData(const ExperimentData &other);
    /**
     * Normal constructor
     * This should only be called by ExperimentDescription
     */
    explicit ExperimentData(const std::string &experiment_name);

};

#endif  // INCLUDE_EXPERIMENTATION_EXPERIMENTDATA_H_
