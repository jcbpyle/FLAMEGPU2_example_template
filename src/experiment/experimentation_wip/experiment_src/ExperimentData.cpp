#include <iostream>

#include "flamegpu/experiment/ExperimentData.h"

/**
 * Constructors
 */
ExperimentData::ExperimentData(const std::string &experiment_name) {
    this.name = experiment_name;
}

ExperimentData::~ExperimentData() { }

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