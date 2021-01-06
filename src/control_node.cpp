#include "ros/ros.h"
#include <geometry_msgs/Twist.h>
#include "geometry_msgs/Vector3.h"
#include <cmath>

#define _USE_MATH_DEFINES

static geometry_msgs::Twist cmd_vel;
static geometry_msgs::Twist prev_vel;


void cmd_Callback(const geometry_msgs::Twist::ConstPtr& msg)
{
    cmd_vel.linear.x = msg->linear.x;
    cmd_vel.angular.z = msg->angular.z;
}


int main(int argc, char **argv)
{

	ros::init(argc, argv, "control_node");
	ros::NodeHandle nh;
	ros::Subscriber imu_sub = nh.subscribe("/control/cmd_vel", 1, cmd_Callback);
	ros::Publisher midterm_control_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
	
	while(ros::ok())
	{	
		ros::Rate loop_rate(10);
		ros::spinOnce();

		if(prev_vel.linear.x != cmd_vel.linear.x)
		{
			midterm_control_pub.publish(cmd_vel);
			std::cout << "state change!!" << std::endl;
		}
		prev_vel = cmd_vel;		
		loop_rate.sleep();
	}
	return 0;
}



	
