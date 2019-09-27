#pragma once
#include "render_interface.h"
#include <camera_spec.h>
#include <memory>
#include <optifuser.h>
#include "camera.h"

class OptifuserRenderer : public IPhysxRenderer, ICameraManager {
public:
  std::map<uint32_t, std::vector<Optifuser::Object *>> mObjectRegistry;
  std::shared_ptr<Optifuser::Scene> mScene;
  Optifuser::GLFWRenderContext *mContext = nullptr;
  Optifuser::FPSCameraSpec cam;
  std::function<GuiInfo(uint32_t)> queryCallback = {};
  std::function<void(uint32_t, const GuiInfo &info)> syncCallback = {};

  // IPhysxRenderer
  virtual void addRigidbody(uint32_t uniqueId, const std::string &meshFile,
                            const physx::PxVec3 &scale) override;
  virtual void addRigidbody(uint32_t uniqueId, physx::PxGeometryType::Enum type,
                            const physx::PxVec3 &scale) override;
  virtual void removeRigidbody(uint32_t uniqueId) override;
  virtual void updateRigidbody(uint32_t uniqueId, const physx::PxTransform &transform) override;

  virtual void bindQueryCallback(std::function<GuiInfo(uint32_t)>) override;
  virtual void bindSyncCallback(std::function<void(uint32_t, const GuiInfo &info)>) override;

public:
  void init();
  void destroy();
  void render();

private:
  std::vector<std::unique_ptr<MountedCamera>> mMountedCameras;

public:
  // ICameraManager
  virtual std::vector<ICamera *> getCameras() override;
  virtual void addCamera(std::string const &name, uint32_t width, uint32_t height, float fovy) override;
};