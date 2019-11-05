//
// Created by sim on 9/26/19.
//

#include "kinematics_articulation_wrapper.h"
#include "simulation.h"
#include <cassert>
#include <iostream>
#include <limits>

namespace sapien {
// Kinematics Articulation Wrapper

KinematicsArticulationWrapper::KinematicsArticulationWrapper(sapien::Simulation &sim) {
  mSimulation = &sim;
}

static std::string jointType2jointTypeString(JointType type) {
  switch (type) {
  case JointType::PRISMATIC: {
    return "prismatic";
  }
  case JointType::REVOLUTE: {
    return "revolute";
  }
  case JointType::FIXED: {
    return "fixed";
  }
  case JointType::CONTINUOUS: {
    return "revolute";
  }
  case JointType::UNDEFINED: {
    return "undefined";
  }
  }
}

KJoint *KinematicsArticulationWrapper::createJoint(const JointType &type, KJoint *parent,
                                                   PxRigidDynamic *link,
                                                   const PxTransform &poseFromParent,
                                                   const PxTransform &poseFromChild,
                                                   PxReal upperLimit, PxReal lowerLimit,
                                                   const std::string &name) {
  std::string qName = name;
  if (!name.empty()) {
    if (jointName2JointPtr.find(name) != jointName2JointPtr.end()) {
      std::cerr << "Duplicated link name: " << name << std::endl;
      throw std::runtime_error("Name duplication");
    }
  } else {
    qName = "undefined_name" + std::to_string(undefinedNameGeneratorId++);
  }

  std::shared_ptr<KJoint> newJoint;
  switch (type) {
  case JointType::REVOLUTE:
    newJoint = std::make_shared<RevoluteKJoint>(link, parent, poseFromChild.getInverse(),
                                                poseFromParent, upperLimit, lowerLimit);
    break;
  case JointType::CONTINUOUS:
    newJoint = std::make_shared<ContinuousKJoint>(link, parent, poseFromChild.getInverse(),
                                                  poseFromParent);
    break;
  case JointType::PRISMATIC:
    newJoint = std::make_shared<PrismaticKJoint>(link, parent, poseFromChild.getInverse(),
                                                 poseFromParent, upperLimit, lowerLimit);
    break;
  case JointType::FIXED:
    newJoint =
        std::make_shared<FixedKJoint>(link, parent, poseFromChild.getInverse(), poseFromParent);
    break;
  case JointType::UNDEFINED:
    std::cerr << "Joint type not support for " << type << std::endl;
    throw std::runtime_error("Unsupported joint");
  }
  if (!parent) {
    mRoot = newJoint.get();
  }
  newJoint->setName(qName);
  jointName2JointPtr[qName] = newJoint;
  return newJoint.get();
}
uint32_t KinematicsArticulationWrapper::dof() const { return DOF; }
void KinematicsArticulationWrapper::buildCache() {
  if (cached) {
    std::cerr << "KinematicsWrapper has already been cached" << std::endl;
    return;
  }
  DOF = 0;
  std::vector<KJoint *> stack = {mRoot->children[0]};
  while (!stack.empty()) {
    auto currentJoint = stack.back();
    stack.pop_back();
    for (auto const &joint : currentJoint->children) {
      stack.push_back(joint);
    }

    // Add link to buffer
    linkListPtr.push_back(currentJoint->childLink);

    // Cache DOF specific information
    uint32_t qDOF = currentJoint->getDof();
    jointStartIndex.push_back(DOF);
    DOF += qDOF;
    std::string name = currentJoint->name;
    for (size_t i = 0; i < qDOF; ++i) {
      jointNameDOF.push_back(name);
      jointLimit.push_back(currentJoint->getLimits()[i]);
    }
    jointDOF.push_back(qDOF);
    jointName.push_back(name);
    jointTypes.push_back(jointType2jointTypeString(currentJoint->type));
    jointListPtr.push_back(jointName2JointPtr[name].get());
  }
  jointNum = jointName2JointPtr.size() - 1;
  jointStartIndex.push_back(DOF);
  driveQvel.resize(DOF, 0);
  qpos.resize(DOF, 0);
  lastStepQpos.resize(DOF, 0);
  qvel.resize(DOF, 0);
  qacc.resize(DOF, 0);
  qf.resize(DOF, 0);
}
std::vector<std::array<PxReal, 2>> KinematicsArticulationWrapper::get_qlimits() const {
  return jointLimit;
}
std::vector<uint32_t> KinematicsArticulationWrapper::get_joint_dofs() const { return jointDOF; }
std::vector<std::string> KinematicsArticulationWrapper::get_joint_names() const {
  return jointName;
}
std::vector<PxReal> KinematicsArticulationWrapper::get_qpos() const { return qpos; }
std::vector<physx::PxReal> KinematicsArticulationWrapper::get_qvel() const { return qvel; }
std::vector<physx::PxReal> KinematicsArticulationWrapper::get_qacc() const { return qacc; }
std::vector<physx::PxReal> KinematicsArticulationWrapper::get_qf() const {
  return std::vector<physx::PxReal>();
}
void KinematicsArticulationWrapper::set_qpos(const std::vector<PxReal> &v) {
  assert(v.size() == DOF);
  size_t j = 0;
  while (j < jointNum) {
    auto start = v.begin() + jointStartIndex[j];
    auto end = v.begin() + jointStartIndex[j + 1];
    std::vector<PxReal> jointQpos(start, end);
    jointListPtr[j]->setQpos(jointQpos);
    j += 1;
  }
  qpos = v;
  hasMagicVelocity = false;
}
void KinematicsArticulationWrapper::set_qvel(const std::vector<PxReal> &v) {
  assert(v.size() == dof());
  hasMagicVelocity = true;
  driveQvel = v;
}
void KinematicsArticulationWrapper::set_qacc(const std::vector<PxReal> &v) {}
void KinematicsArticulationWrapper::set_qf(const std::vector<PxReal> &v) {}
void KinematicsArticulationWrapper::set_drive_target(const std::vector<PxReal> &v) {
  mRoot->childLink->setKinematicTarget(rootPose);
  assert(v.size() == DOF);
  size_t j = 0;
  while (j < jointNum) {
    auto start = v.begin() + jointStartIndex[j];
    auto end = v.begin() + jointStartIndex[j + 1];
    std::vector<PxReal> jointQpos(start, end);
    jointListPtr[j]->driveQpos(jointQpos);
    j += 1;
  }
  qpos = v;
}
std::vector<std::string> KinematicsArticulationWrapper::get_qnames() const { return jointNameDOF; }

// Update function should be called in the simulation loop
void KinematicsArticulationWrapper::update(PxReal timestep) {
  // Update drive target based on controllers
  if (hasMagicVelocity) {
    std::vector<PxReal> driveQpos = qpos;
    for (std::size_t i = 0; i < dof(); ++i) {
      // Update drive of next step based on the drive stepSize
      PxReal newQ = qpos[i] + driveQvel[i] * timestep;
      auto [upperLimit, lowerLimit] = jointLimit[i];
      newQ = newQ > upperLimit ? upperLimit : newQ;
      driveQpos[i] = newQ < lowerLimit ? lowerLimit : newQ;
    }
    set_drive_target(driveQpos);
    qpos = driveQpos;
  }

  // Update stepSize based on time interval if the set_drive_target function
  for (size_t i = 0; i < dof(); ++i) {
    qvel[i] = (qpos[i] - lastStepQpos[i]) / timestep;
  }
  lastStepQpos = qpos;
}
EArticulationType KinematicsArticulationWrapper::get_articulation_type() const {
  return EArticulationType::KINEMATIC_ARTICULATION;
}
void KinematicsArticulationWrapper::move_base(const PxTransform &T) {
  rootPose = T;
  set_drive_target(qpos);
}
std::vector<std::string> KinematicsArticulationWrapper::get_link_names() const {
  std::vector<std::string> names;
  for (auto l : linkListPtr) {
    names.push_back(l->getName());
  }
  return names;
}

std::vector<physx_id_t> KinematicsArticulationWrapper::get_link_ids() const {
  std::vector<physx_id_t> segmentation_ids;
  for (auto l : linkListPtr) {
    segmentation_ids.push_back(mSimulation->mActor2LinkId[l]);
  }
  return segmentation_ids;
}

std::vector<int> KinematicsArticulationWrapper::get_link_joint_indices() const {
  std::vector<int> jointIndices;
  for (uint32_t i = 0; i < linkListPtr.size(); ++i) {
    jointIndices.push_back(i);
  }
  return jointIndices;
}

physx::PxTransform KinematicsArticulationWrapper::get_link_joint_pose(uint32_t idx) const {
  assert(idx < linkListPtr.size());
  return linkListPtr[idx]->getGlobalPose() * jointListPtr[idx]->getPoseToChild().getInverse();
}
std::vector<std::string> KinematicsArticulationWrapper::get_joint_types() const {
  return jointTypes;
}

} // namespace sapien
