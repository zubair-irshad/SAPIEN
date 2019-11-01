//
// Created by sim on 10/31/19.
//
#include "actor_builder.h"
#include "articulation_builder.h"
#include "controller/controller_manger.h"
#include "device/single_gripper_ps3.h"
#include "optifuser_renderer.h"
#include "simulation.h"
#include <optifuser.h>
#include <thread>

using namespace sapien;

void run() {
  Renderer::OptifuserRenderer renderer;

  renderer.cam.position = {0.5, -4, 0.5};
  renderer.cam.setForward({0, 1, 0});
  renderer.cam.setUp({0, 0, 1});

  renderer.setAmbientLight({.4, .4, .4});
  renderer.setShadowLight({1, -1, -1}, {.5, .5, .5});
  renderer.addPointLight({2, 2, 2}, {1, 1, 1});
  renderer.addPointLight({2, -2, 2}, {1, 1, 1});
  renderer.addPointLight({-2, 0, 2}, {1, 1, 1});

  Simulation sim;
  sim.setRenderer(&renderer);
  sim.setTimestep(1.f / 300.f);
  sim.addGround(0.0);
  auto loader = sim.createURDFLoader();
  loader->fixLoadedObject = true;
  std::string partFile = "/home/sim/project/mobility_convex/45091/mobility.urdf";
  loader->load(partFile)->articulation->teleportRootLink({{1.0, 0.3, 0.8}, PxIdentity}, true);

  loader->balancePassiveForce = true;
  auto gripperMaterial = sim.mPhysicsSDK->createMaterial(5, 6, 0.01);
  auto wrapper = loader->load("../assets/robot/single_hand.urdf", gripperMaterial);
  wrapper->set_drive_property(200, 40, 20, {0, 1, 2, 3, 4, 5});
  wrapper->set_drive_property(1, 0.05, 1, {6, 7, 8});
  wrapper->set_qpos({0, 0, 1, 0, 0, 0, 0, 0, 0});
  wrapper->set_drive_target({0.641, 0.789, 0.218, -1.8, 0.08, 0, 0, 0, 0});

  auto controllableWrapper = sim.createControllableArticulationWrapper(wrapper);
  auto manger = std::make_unique<robot::ControllerManger>("kg3", controllableWrapper);
  robot::KinovaGripperPS3 ps3(manger.get());
  ps3.set_translation_velocity(0.15);
  ps3.set_gripper_velocity(1);

  renderer.showWindow();
  std::vector<std::vector<PxReal>> temp;
  while (true) {
    sim.step();
    sim.updateRenderer();
    renderer.render();
    ps3.step();
    //    temp.push_back(sim.dump());

    auto gl_input = Optifuser::getInput();
    if (gl_input.getKeyState(GLFW_KEY_Q)) {
      break;
    }
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "ps3_movo");
  run();
  return 0;
}