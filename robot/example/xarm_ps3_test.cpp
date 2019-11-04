#include "actor_builder.h"
#include "articulation_builder.h"
#include "controller/controller_manger.h"
#include "device/xarm6_ps3.h"
#include "optifuser_renderer.h"
#include "simulation.h"
#include <optifuser.h>
#include <thread>

using namespace sapien;

void run() {
  Renderer::OptifuserRenderer renderer;
  renderer.cam.position = {0, -2, 0.5};
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
  loader->balancePassiveForce = true;

  auto builder = sim.createActorBuilder();
  builder->addBoxShape({{0, 0, 0}, PxIdentity}, {0.5, 1.5, 0.3});
  builder->addBoxVisual({{0, 0, 0}, PxIdentity}, {0.5, 1.5, 0.3});
  auto actor = builder->build(false, false, "test", true);
  actor->setGlobalPose({{2.0, 0.3, 0.3}, PxIdentity});

  auto builder1 = sim.createActorBuilder();
  builder1->addObjVisual("../assets/object/029_plate/google_16k/textured.dae");
  builder1->addConvexShapeFromObj("../assets/object/029_plate/google_16k/textured.obj");
  auto plate = builder1->build(false, false, "plate", true);
  plate->setGlobalPose({{2.0, 0.3, 2}, PxIdentity});

  //  loader->load("../assets/46627/test.urdf")
  //      ->articulation->teleportRootLink({{2.0, 5.3, 0.4}, PxIdentity}, true);
  auto wrapper = loader->load("../assets/robot/xarm6.urdf");
  wrapper->set_drive_property(2000, 500);

  auto controllableWrapper = sim.createControllableArticulationWrapper(wrapper);
  auto manger = std::make_unique<robot::ControllerManger>("xarm6", controllableWrapper);
  auto ps3 = robot::XArm6PS3(manger.get());

  wrapper->set_qpos({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  wrapper->set_drive_target({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

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