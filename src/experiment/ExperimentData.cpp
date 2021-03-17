#include <iostream>

#include "experimentation/ExperimentData.h"

#include "flamegpu/model/AgentData.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/AgentFunctionData.h"
#include "flamegpu/model/LayerData.h"
#include "flamegpu/model/SubModelData.h"
#include "flamegpu/model/SubAgentData.h"
#include "flamegpu/model/SubEnvironmentData.h"
#include "flamegpu/runtime/HostFunctionCallback.h"

const char *ExperimentData::DEFAULT_STATE = "default";

/**
 * Constructors
 */
ExperimentData::ExperimentData(const std::string &experiment_name)
    : name(model_name) { }

ExperimentData::~ExperimentData() { }

ExperimentData::ExperimentData(const ExperimentData &other)
    : name(other.name) {
    // Must be called from clone() so that items are all init
}

bool ExperimentData::operator==(const ExperimentData& rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    if (name == rhs.name) {
            return true;
    }
    return false;
}

bool ExperimentData::operator!=(const ExperimentData& rhs) const {
    return !operator==(rhs);
}