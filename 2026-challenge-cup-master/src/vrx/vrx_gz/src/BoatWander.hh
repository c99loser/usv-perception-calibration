/*
 * BoatWander — gz-sim 系统插件
 *
 * 让障碍船(roboboat)在起始点附近随机游走(差速推进器控制)。
 * 完全在仿真引擎内部运行: 每个仿真步读模型自身位姿, 计算朝目标的
 * 推进器转速命令, 通过 JointVelocityCmd 直接驱动螺旋桨关节。
 * 零话题依赖, 零外部节点, gazebo 加载模型即自动游走。
 *
 * SDF 配置(加到 model 的 <plugin>):
 *   <plugin filename="libBoatWander.so" name="vrx::BoatWander">
 *     <wander_radius>15</wander_radius>
 *     <thrust_speed>30</thrust_speed>
 *     <turn_speed>15</turn_speed>
 *   </plugin>
 */

#ifndef VRX_BOATWANDER_HH_
#define VRX_BOATWANDER_HH_

#include <gz/sim/System.hh>
#include <gz/sim/Entity.hh>

#include <memory>
#include <random>
#include <string>

namespace vrx
{
  class BoatWander :
    public gz::sim::System,
    public gz::sim::ISystemConfigure,
    public gz::sim::ISystemPreUpdate
  {
    public: BoatWander();
    public: ~BoatWander() override = default;

    public: void Configure(const gz::sim::Entity &_entity,
                           const std::shared_ptr<const sdf::Element> &_sdf,
                           gz::sim::EntityComponentManager &_ecm,
                           gz::sim::EventManager &_eventMgr) override;

    public: void PreUpdate(const gz::sim::UpdateInfo &_info,
                           gz::sim::EntityComponentManager &_ecm) override;

    private: class Implementation;
    private: std::unique_ptr<Implementation> dataPtr;
  };
}

#endif
