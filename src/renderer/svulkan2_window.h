#pragma once
#include "svulkan2/ui/ui.h"
#include "svulkan2_renderer.h"

namespace sapien {
namespace Renderer {

#ifdef _DEBUG
class FPSCameraControllerDebug {
  svulkan2::scene::Node *mCamera;
  glm::vec3 mRPY{0, 0, 0};
  glm::vec3 mXYZ{0, 0, 0};

  glm::vec3 mForward;
  glm::vec3 mUp;
  glm::vec3 mLeft;

  glm::quat mInitialRotation;

  void update();

public:
  FPSCameraControllerDebug(svulkan2::scene::Node &node, glm::vec3 const &forward,
                           glm::vec3 const &up);

  void setRPY(float roll, float pitch, float yaw);
  void setXYZ(float x, float y, float z);

  inline glm::vec3 getRPY() const { return mRPY; };
  inline glm::vec3 getXYZ() const { return mXYZ; };

  void move(float forward, float left, float up);
  void rotate(float roll, float pitch, float yaw);
};
#endif

class SVulkan2Window {
  SVulkan2Renderer *mRenderer{};
  std::string mShaderDir{};
  std::unique_ptr<svulkan2::renderer::Renderer> mSVulkanRenderer{};
  SVulkan2Scene *mScene{};

  std::unique_ptr<svulkan2::renderer::GuiWindow> mWindow;

  vk::UniqueSemaphore mSceneRenderSemaphore;
  vk::UniqueFence mSceneRenderFence;
  vk::UniqueCommandBuffer mCommandBuffer;

  bool mClosed{};

#ifdef _DEBUG
  std::unique_ptr<FPSCameraControllerDebug> mCameraController{};
#endif
public:
  SVulkan2Window(SVulkan2Renderer &renderer, std::string const &shaderDir);
  ~SVulkan2Window();
  void setScene(SVulkan2Scene *scene);
  void setCameraParameters(float near, float far, float fovy);
  void setCameraPosition(glm::vec3 const &pos);
  void setCameraRotation(glm::quat const &rot);

  std::vector<std::string> getDisplayTargetNames() const;
  void render(std::string const &targetName,
              std::vector<std::shared_ptr<svulkan2::ui::Window>> uiWindows = {});
  void show();
  void hide();

  void close();

  bool windowCloseRequested();

  std::tuple<std::vector<float>, std::array<uint32_t, 3>>
  downloadFloatTarget(std::string const &name);
  std::tuple<std::vector<uint32_t>, std::array<uint32_t, 3>>
  downloadUint32Target(std::string const &name);
  std::tuple<std::vector<uint8_t>, std::array<uint32_t, 3>>
  downloadUint8Target(std::string const &name);

  std::vector<float> downloadFloatTargetPixel(std::string const &name, uint32_t x, uint32_t y);
  std::vector<uint32_t> downloadUint32TargetPixel(std::string const &name, uint32_t x, uint32_t y);
  std::vector<uint8_t> downloadUint8TargetPixel(std::string const &name, uint32_t x, uint32_t y);

  bool isKeyDown(std::string const &key);
  bool isKeyPressed(std::string const &key);
  bool isMouseKeyDown(int key);
  bool isMouseKeyClicked(int key);
  std::array<float, 2> getMousePosition();
  std::array<float, 2> getMouseDelta();
  std::array<float, 2> getMouseWheelDelta();

private:
  svulkan2::scene::Camera *getCamera();
};

} // namespace Renderer
} // namespace sapien