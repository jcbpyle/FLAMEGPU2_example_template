#include "flamegpu/experiment/ExperimentDescription.h"

/**
* Constructors
*/
ExperimentDescription::ExperimentDescription(const std::string &experiment_name)
    : experiment_pointer(new ExperimentData(experiment_name)) { }

bool ExperimentDescription::operator==(const ExperimentDescription& rhs) const {
    return *this->experiment_pointer == *rhs.experiment_pointer;  // Compare content is functionally the same
}
bool ExperimentDescription::operator!=(const ExperimentDescription& rhs) const {
    return !(*this == rhs);
}

/**
* Const Accessors
*/
std::string ExperimentDescription::getName() const {
    return experiment_pointer->name;
}
