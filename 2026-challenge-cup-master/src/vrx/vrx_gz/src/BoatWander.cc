/*
 * BoatWander 实现 — 见 BoatWander.hh
 */
#include <gz/msgs/config.hh>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/double.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>

#include <mutex>

#include <gz/sim/components.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>

#include "BoatWander.hh"

using namespace gz;
using namespace sim;
using namespace vrx;

//////////////////////////////////////////////////
class vrx::BoatWander::Implementation
{
  /// \brief 模型接口
  public: Model model{kNullEntity};

  /// \brief gz transport 节点(发布 cmd_thrust + 订阅位姿)
  public: std::shared_ptr<transport::Node> gzNode;
  public: transport::Node::Publisher leftPub;
  public: transport::Node::Publisher rightPub;
  public: bool pubsCreated{false};
  public: std::string worldName{"berthing_test"};

  /// \brief 关节名(用于构建话题名)
  public: std::string leftJointName{"left_engine_propeller_joint"};
  public: std::string rightJointName{"right_engine_propeller_joint"};

  /// \brief 模型名
  public: std::string modelName;

  /// \brief 实时位姿(gz transport 回读, 线程安全)
  public: std::mutex mtx;
  public: double curX{0}, curY{0}, curYaw{0};
  public: bool poseValid{false};

  /// \brief 游走参数
  public: double wanderRadius{15.0};
  public: double thrustSpeed{30.0};   // 螺旋桨转速(正向)
  public: double turnSpeed{15.0};     // 转向转速差

  /// \brief 起始位置(游走中心)
  public: double cx{0}, cy{0};
  public: bool initialized{false};

  /// \brief 当前目标
  public: double targetX{0}, targetY{0};
  public: double lastSwitch{0.0};

  /// \brief 随机数生成器
  public: std::mt19937 rng{std::random_device{}()};
  public: std::uniform_real_distribution<double> angleDist{0, 2 * M_PI};
  public: std::uniform_real_distribution<double> radiusDist{0.2, 0.6};
  public: std::uniform_real_distribution<double> switchDist{5.0, 10.0};

  /// \brief 选新目标(在 0.1-0.3 倍半径内, 大幅留余量给惯性滑行)
  public: void NewTarget()
  {
    std::uniform_real_distribution<double> smallRadius{0.1, 0.3};
    double ang = angleDist(rng);
    double r = wanderRadius * smallRadius(rng);
    targetX = cx + r * cos(ang);
    targetY = cy + r * sin(ang);
  }

  /// \brief 选新目标
};

//////////////////////////////////////////////////
BoatWander::BoatWander()
  : dataPtr(std::make_unique<Implementation>())
{
}

//////////////////////////////////////////////////
void BoatWander::Configure(const Entity &_entity,
                           const std::shared_ptr<const sdf::Element> &_sdf,
                           EntityComponentManager &_ecm,
                           EventManager & /*_eventMgr*/)
{
  this->dataPtr->model = Model(_entity);

  // 读参数
  if (_sdf->HasElement("wander_radius"))
    this->dataPtr->wanderRadius = _sdf->Get<double>("wander_radius");
  if (_sdf->HasElement("thrust_speed"))
    this->dataPtr->thrustSpeed = _sdf->Get<double>("thrust_speed");
  if (_sdf->HasElement("turn_speed"))
    this->dataPtr->turnSpeed = _sdf->Get<double>("turn_speed");

  std::string leftName = "left_engine_propeller_joint";
  std::string rightName = "right_engine_propeller_joint";
  if (_sdf->HasElement("left_joint"))
    leftName = _sdf->Get<std::string>("left_joint");
  if (_sdf->HasElement("right_joint"))
    rightName = _sdf->Get<std::string>("right_joint");
  this->dataPtr->leftJointName = leftName;
  this->dataPtr->rightJointName = rightName;
  this->dataPtr->modelName = this->dataPtr->model.Name(_ecm);
  // world名从父元素读
  if (_sdf->HasElement("world"))
    this->dataPtr->worldName = _sdf->Get<std::string>("world");

  // 从 gz transport 订阅 dynamic_pose 获取实时位姿
  this->dataPtr->gzNode = std::make_shared<transport::Node>();
  // 订阅自己的位姿(dynamic_pose/info 含所有model位姿)
  this->dataPtr->gzNode->Subscribe<gz::msgs::Pose_V>(
      "/world/" + this->dataPtr->worldName + "/dynamic_pose/info",
      [this](const gz::msgs::Pose_V &_msg) {
        for (const auto &p : _msg.pose())
        {
          if (p.name() == this->dataPtr->modelName)
          {
            std::lock_guard<std::mutex> lk(this->dataPtr->mtx);
            this->dataPtr->curX = p.position().x();
            this->dataPtr->curY = p.position().y();
            this->dataPtr->curYaw = 2.0 * std::atan2(
                p.orientation().z(), p.orientation().w());
            this->dataPtr->poseValid = true;
            if (!this->dataPtr->initialized)
            {
              this->dataPtr->cx = p.position().x();
              this->dataPtr->cy = p.position().y();
              this->dataPtr->NewTarget();
              this->dataPtr->initialized = true;
              std::cerr << "[BoatWander] " << this->dataPtr->modelName
                        << " 初始中心: (" << this->dataPtr->cx << ", "
                        << this->dataPtr->cy << ")" << std::endl;
            }
          }
        }
      });

  // 创建 gz transport publisher (cmd_thrust, Thruster插件读取的接口)
  std::string leftTopic = "/model/" + this->dataPtr->modelName +
      "/joint/" + leftName + "/cmd_thrust";
  std::string rightTopic = "/model/" + this->dataPtr->modelName +
      "/joint/" + rightName + "/cmd_thrust";
  this->dataPtr->leftPub = this->dataPtr->gzNode->Advertise<gz::msgs::Double>(leftTopic);
  this->dataPtr->rightPub = this->dataPtr->gzNode->Advertise<gz::msgs::Double>(rightTopic);
  this->dataPtr->pubsCreated = true;

  std::cerr << "[BoatWander] 配置: model=" << this->dataPtr->modelName
            << " radius=" << this->dataPtr->wanderRadius
            << " thrust=" << this->dataPtr->thrustSpeed << std::endl;
}

//////////////////////////////////////////////////
void BoatWander::PreUpdate(const UpdateInfo &_info,
                           EntityComponentManager & /*_ecm*/)
{
  if (!this->dataPtr->pubsCreated || !this->dataPtr->poseValid)
    return;

  // 从 gz transport 回读位姿(线程安全)
  double x, y, yaw;
  {
    std::lock_guard<std::mutex> lk(this->dataPtr->mtx);
    x = this->dataPtr->curX;
    y = this->dataPtr->curY;
    yaw = this->dataPtr->curYaw;
  }

  double now = std::chrono::duration<double>(
      _info.simTime).count();
  double dx = this->dataPtr->targetX - x;
  double dy = this->dataPtr->targetY - y;
  double dc = sqrt((x - this->dataPtr->cx) * (x - this->dataPtr->cx) +
                   (y - this->dataPtr->cy) * (y - this->dataPtr->cy));

  // 范围约束: 超出0.5*radius → 目标设为中心(软约束, 转向回拉)
  bool returnMode = dc > this->dataPtr->wanderRadius * 0.5;
  if (returnMode)
  {
    this->dataPtr->targetX = this->dataPtr->cx;
    this->dataPtr->targetY = this->dataPtr->cy;
  }
  else if (now - this->dataPtr->lastSwitch >
               this->dataPtr->switchDist(this->dataPtr->rng) &&
           dc < this->dataPtr->wanderRadius * 0.5)
  {
    this->dataPtr->NewTarget();
    this->dataPtr->lastSwitch = now;
  }

  // 硬约束: 超出 radius → 用 set_pose 强制拉回到 0.5*radius 处(防止惯性滑出)
  // 只在超出时触发, 正常游走不触发
  if (dc > this->dataPtr->wanderRadius)
  {
    double ang = atan2(y - this->dataPtr->cy, x - this->dataPtr->cx);
    double pullR = this->dataPtr->wanderRadius * 0.4;
    double nx = this->dataPtr->cx + pullR * cos(ang);
    double ny = this->dataPtr->cy + pullR * sin(ang);
    gz::msgs::Pose poseReq;
    poseReq.set_name(this->dataPtr->modelName);
    poseReq.mutable_position()->set_x(nx);
    poseReq.mutable_position()->set_y(ny);
    poseReq.mutable_position()->set_z(0.2);
    poseReq.mutable_orientation()->set_w(1.0);
    std::string setPoseSvc = "/world/" + this->dataPtr->worldName + "/set_pose";
    this->dataPtr->gzNode->Request(setPoseSvc, poseReq);
    // 更新本地估计
    {
      std::lock_guard<std::mutex> lk(this->dataPtr->mtx);
      this->dataPtr->curX = nx;
      this->dataPtr->curY = ny;
    }
    return;  // 拉回后跳过本帧推力
  }

  // 差速控制(始终正向推力, 朝目标转向)
  double targetYaw = atan2(dy, dx);
  double yawErr = atan2(sin(targetYaw - yaw), cos(targetYaw - yaw));
  double turn = yawErr / M_PI * 2.0;
  if (turn > 1.0) turn = 1.0;
  if (turn < -1.0) turn = -1.0;

  double leftVel = this->dataPtr->thrustSpeed - turn * this->dataPtr->turnSpeed;
  double rightVel = this->dataPtr->thrustSpeed + turn * this->dataPtr->turnSpeed;
  // 拉回模式: 加大前进推力
  if (returnMode)
  {
    leftVel = this->dataPtr->thrustSpeed * 1.5 - turn * this->dataPtr->turnSpeed;
    rightVel = this->dataPtr->thrustSpeed * 1.5 + turn * this->dataPtr->turnSpeed;
  }

  gz::msgs::Double leftMsg, rightMsg;
  leftMsg.set_data(leftVel);
  rightMsg.set_data(rightVel);
  this->dataPtr->leftPub.Publish(leftMsg);
  this->dataPtr->rightPub.Publish(rightMsg);
}

// 注册插件
GZ_ADD_PLUGIN(vrx::BoatWander,
              gz::sim::System,
              gz::sim::ISystemConfigure,
              gz::sim::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(vrx::BoatWander, "vrx::BoatWander")
