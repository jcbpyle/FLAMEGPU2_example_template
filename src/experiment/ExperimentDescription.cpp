#include "experimentation/ExperimentDescription.h"
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/LayerDescription.h"
#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/runtime/messaging/BruteForce.h"
#include "flamegpu/model/SubModelData.h"
#include "flamegpu/model/SubModelDescription.h"

/**
* Constructors
*/
ExperimentDescription::ExperimentDescription(const std::string &experiment_name)
    : experiment(new ExperimentData(experiment_name)) { }

bool ExperimentDescription::operator==(const ExperimentDescription& rhs) const {
    return *this->experiment == *rhs.experiment;  // Compare content is functionally the same
}
bool ExperimentDescription::operator!=(const ExperimentDescription& rhs) const {
    return !(*this == rhs);
}

/**
* Const Accessors
*/
std::string ExperimentDescription::getName() const {
    return experiment->name;
}
