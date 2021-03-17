#ifndef INCLUDE_FLAMEGPU_EXPERIMENT_EXPERIMENTDATA_H_
#define INCLUDE_FLAMEGPU_EXPERIMENT_EXPERIMENTDATA_H_

#include <unordered_map>
#include <list>
#include <memory>
#include <typeindex>
#include <set>
#include <string>

/**
 * This is the internal data store for ExperimentDescription
 * Users should only access that data stored within via an instance of ExperimentDescription
 */
class ExperimentData : std::enable_shared_from_this<ExperimentData>{
    //virtual ~ExperimentData();
    /**
     * Description needs full access
     */
    friend class ExperimentDescription;    
    
    /**
     * The name of the experiment
     * This must be unique among Experiment instances
     */
    std::string name;

    /**
     * Equality operator, checks whether ExperimentData hierarchies are functionally the same
     * @returns True when experiments are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const ExperimentData& rhs) const;
    /**
     * Equality operator, checks whether ExperimentData hierarchies are functionally different
     * @returns True when experiments are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const ExperimentData& rhs) const;


 protected:
    /**
     * Normal constructor
     * This should only be called by ExperimentDescription
     */
    explicit ExperimentData(const std::string &experiment_name);

};

#endif  // INCLUDE_FLAMEGPU_EXPERIMENT_EXPERIMENTDATA_H_
