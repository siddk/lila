/**
 * lilaVelocityControl.cpp
 *
 * Low-Level Libfranka Code for Controlling the Robot using Joint Velocity Commands; the Robot sends back Position,
 * Velocity, Torque, and the Jacobian. Additionally supports Gripper Manipulation.
 *
 */

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/model.h>
#include <franka/robot.h>
#include <iostream>
#include <sstream>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

// Create SharedMemory Struct
struct SharedMemory {
    volatile std::sig_atomic_t* doRun = nullptr;
};

// Create Signal Handling Variable
volatile std::sig_atomic_t g_doRun = true;

// Define LILAVelocityController (Methods, Fields)
class LILAVelocityController {
  public:
    LILAVelocityController(franka::Model& model, int port);
    franka::JointVelocities operator()(const franka::RobotState& robot_state, franka::Duration period);

  private:
    franka::Model* modelPtr;

    // Control Parameters
    double time = 0.0;
    std::array<double, 100> buffer1;
    std::array<double, 100> buffer2;
    std::array<double, 100> buffer3;
    std::array<double, 100> buffer4;
    std::array<double, 100> buffer5;
    std::array<double, 100> buffer6;
    std::array<double, 100> buffer7;
    std::array<double, 100> bufferRate;
    std::array<double, 7> qdot;
    std::array<double, 7> control_input;
    double MovingAverage(std::array<double, 100>& buffer, double input);

    // Socket Parameters
    int sock = 0;
    int valread;
    struct sockaddr_in serv_addr;
    char buffer[200] = {0};

    // Non-Blocking Reading
    int steps = 0;
};

// Constructor - Takes Robot Model*, Gripper*, and Port
LILAVelocityController::LILAVelocityController(franka::Model& model, int port) {
  // Store Model and Gripper Pointer
  modelPtr = &model;

  for (int i = 0; i < 7; i++) {
    control_input[i] = 0.0;
  }

  for (int i = 0; i < 100; i++) {
    buffer1[i] = 0.0;
    buffer2[i] = 0.0;
    buffer3[i] = 0.0;
    buffer4[i] = 0.0;
    buffer5[i] = 0.0;
    buffer6[i] = 0.0;
    buffer7[i] = 0.0;
    bufferRate[i] = 0.0;
  }

  // Create Connection to Socket
  if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    std::cout << "Socket Creation Error!" << std::endl;
  }
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);
  if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
    std::cout << "Invalid Address / Address Not Supported" << std::endl;
  }
  if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    std::cout << "Connection Failed" << std::endl;
  }
  int status = fcntl(sock, F_SETFL, fcntl(sock, F_GETFL, 0) | O_NONBLOCK);
  if (status == -1) {
    std::cout << "Failed to Make the Socket Non-Blocking" << std::endl;
  }
}

// Implement a MovingAverage Function for Smoothing Input Velocities (from Velocity Controller)
double LILAVelocityController::MovingAverage(std::array<double, 100>& buffer, double input) {
  double filtered_input = 0.0;
  for (int i = 0; i < 100; i++) {
    filtered_input = filtered_input + buffer[i];
  }
  filtered_input = filtered_input / 100.0;
  for (int i = 0; i < 99; i++) {
    buffer[i] = buffer[i + 1];
  }
  buffer[99] = input;
  return filtered_input;
}

franka::JointVelocities LILAVelocityController::operator()(const franka::RobotState& robot_state, franka::Duration period) {
  time += period.toSec();
  std::array<double, 7> joint_position = robot_state.q;
  std::array<double, 7> joint_velocity = robot_state.dq;
  std::array<double, 7> applied_torque = robot_state.tau_ext_hat_filtered;
  std::array<double, 42> jacobian = modelPtr->zeroJacobian(franka::Frame::kEndEffector, robot_state);
  double send_rate = robot_state.control_command_success_rate;

  // Assemble "State" Representation (to be sent over Socket)
  std::string state = "s,";
  for (int i = 0; i < 7; i++) {
    state.append(std::to_string(joint_position[i]));
    state.append(",");
  }

  // Add Joint Velocities
  for (int i = 0; i < 7; i++) {
    state.append(std::to_string(joint_velocity[i]));
    state.append(",");
  }

  // Add Joint Torque
  for (int i = 0; i < 7; i++) {
    state.append(std::to_string(applied_torque[i]));
    state.append(",");
  }

  // Add Jacobian
  for (int i = 0; i < 42; i++) {
    state.append(std::to_string(jacobian[i]));
    state.append(",");
  }

  // Wrap Message with a Termination Character
  char cstr[state.size() + 1];
  std::copy(state.begin(), state.end(), cstr);
  cstr[state.size()] = '\0';

  // Read & Write to Socket
  if (steps % 5 < 1) {
    valread = read(sock, buffer, 200);
    send(sock, cstr, strlen(cstr), 0);
    if (valread > 0) {
      std::stringstream ss(buffer);
      bool doVelocity = false;

      while (not doVelocity) {
        std::string substr;
        getline(ss, substr, ',');
        if ( substr[0] == 's') {
          doVelocity = true;
        }
      }

      for (int i = 0; i < 7; i++) {
        std::string substr;
        getline(ss, substr, ',');
        double term = std::stod(substr);
        control_input[i] = term;
      }
    } else {
      for (int i = 0; i < 7; i++) {
        control_input[i] = 0.000001;
      }
    }
  }

  double qdot1 = MovingAverage(buffer1, control_input[0]);
  double qdot2 = MovingAverage(buffer2, control_input[1]);
  double qdot3 = MovingAverage(buffer3, control_input[2]);
  double qdot4 = MovingAverage(buffer4, control_input[3]);
  double qdot5 = MovingAverage(buffer5, control_input[4]);
  double qdot6 = MovingAverage(buffer6, control_input[5]);
  double qdot7 = MovingAverage(buffer7, control_input[6]);
  double comm_rate = MovingAverage(bufferRate, send_rate);
  qdot = {{qdot1, qdot2, qdot3, qdot4, qdot5, qdot6, qdot7}};
  franka::JointVelocities velocity(qdot);

  steps = steps + 1;
  if (steps % 1000 < 1) {
    std::cout << "control_command_success_rate: " << comm_rate << std::endl;
  }

  return velocity;
}

void gripperController(std::string robotIP, int doHoming, int port, std::shared_ptr<SharedMemory> shared) {
    // Connect to Gripper & Perform a Homing
    franka::Gripper gripper(robotIP);
    if (doHoming == 1) {
        if (!gripper.homing()) {
            std::cerr << "Check the Gripper, something is wrong!" << std::endl;
        }
    }
    std::cout << "Gripper is in business, starting thread..." << std::endl;

    // Gripper State Variables
    franka::Gripper* gripperPtr = &gripper;
    bool gripperOpen = true;

    // Socket Parameters
    int sock = 0;
    int valread;
    struct sockaddr_in serv_addr;
    char buffer[200] = {0};

    // Create Connection to Socket
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      std::cout << "Socket Creation Error!" << std::endl;
    }
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
      std::cout << "Invalid Address / Address Not Supported" << std::endl;
    }
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
      std::cout << "Connection Failed" << std::endl;
    }
    int status = fcntl(sock, F_SETFL, fcntl(sock, F_GETFL, 0) | O_NONBLOCK);
    if (status == -1) {
      std::cout << "Failed to Make the Socket Non-Blocking" << std::endl;
    }

    // Loop While Running
    try {
        while (*shared->doRun) {
            // Sleep at 1000 Hz (Libfranka Frequency)
            sleep(0.001);

            // Attempt to read from the socket
            valread = read(sock, buffer, 200);
            if (valread > 0) {
                std::stringstream ss(buffer);
                bool fireGripper = false;

                while (not fireGripper) {
                    std::string substr;
                    getline(ss, substr, ',');
                    if ( substr[0] == 'g' ) {
                        fireGripper = true;
                    }
                }

                // Control Gripper Logic
                if (fireGripper) {
                  // Gripper is Open...
                  if (gripperOpen) {
                    // Width of 0.0 is Closed, the speed, force are defaults from:
                    //    https://frankaemika.github.io/libfranka/grasp_object_8cpp-example.html
                    gripperPtr->grasp(0.00001, 0.2, 120);
                    gripperOpen = false;
                  } else {
                    // Width of 0.0857 is Open, the speed, force are defaults from:
                    //    https://frankaemika.github.io/libfranka/grasp_object_8cpp-example.html
                    gripperPtr->grasp(0.08570, 0.2, 120);
                    gripperOpen = true;
                  }
                }
            }
        }
    } catch (const std::exception& e) {
      std::cerr << "GripperController(): " << e.what() << std::endl;
    }
    std::cout << "GripperController(): Exiting..." << std::endl;
}

int main(int argc, char** argv) {
  // Use Default Arguments
  std::string robotIP;
  int controlPort;
  int gripperPort;
  int doHoming;
  std::thread gripperThread;

  // Argument Parsing Logic
  if (argc < 3) {
    doHoming = std::stoi(argv[1]);
    robotIP = "172.16.0.2";
    controlPort = 8080;
    gripperPort = controlPort + 1;
  } else if (argc < 4) {
    doHoming = std::stoi(argv[1]);
    robotIP = argv[2];
    controlPort = 8080;
    gripperPort = controlPort + 1;
  } else if (argc == 4) {
    doHoming = std::stoi(argv[1]);
    robotIP = argv[2];
    controlPort = std::stoi(argv[3]);
    gripperPort = controlPort + 1;
  } else {
    std::cerr << "Usage: " << std::endl
              << "Do Gripper Homing [0 or 1]" << std::endl
              << "IP of robot" << std::endl
              << "Port for socket channel" << std::endl
              << "Example IP is [172.16.0.2]." << std::endl
              << "Example Port is [8080]." << std::endl
              << "Note Port for Gripper is 1 + PORT [8081]." << std::endl;
    return -1;
  }

  try {
    // Instantiate Robot & Underlying Robot Model
    franka::Robot robot(robotIP);
    franka::Model model = robot.loadModel();
    robot.automaticErrorRecovery();

    // Create Shared Memory
    auto shared = std::make_shared<SharedMemory>();
    shared->doRun = &g_doRun;

    // Start Gripper Thread
    gripperThread = std::thread(gripperController, robotIP, doHoming, gripperPort, shared);

    // Create Velocity Controller
    LILAVelocityController motion_generator(model, controlPort);
    robot.setCollisionBehavior(
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}});
    robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
    robot.control(motion_generator);

  } catch (franka::Exception const& e) {
    std::cout << e.what() << std::endl;
    std::cout << "Stopped VelocityControl Loop." << std::endl;
    g_doRun = false;
  }

  if (gripperThread.joinable()) {
    std::cout << "Waiting for Gripper Thread to Terminate..." << std::endl;
    gripperThread.join();
  }

  return 0;
}
