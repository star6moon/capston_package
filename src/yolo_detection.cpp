#include "ros/ros.h"
#include <std_msgs/Float64.h>
#include <cmath>

#define _USE_MATH_DEFINES

std_msgs::Float64 yolo_sign;
std_msgs::Float64 prev_yolo_sign;

int main(int argc, char **argv)
{
	ros::init(argc, argv, "yolo_detection");
	ros::NodeHandle nh;
	ros::Publisher yolo_pub = nh.advertise<std_msgs::Float64>("/cognition/yolo", 1);
	
	while(ros::ok())
	{	
		ros::Rate loop_rate(10);
		ros::spinOnce();

        yolo_sign.data = 0;

		if(prev_yolo_sign.data != yolo_sign.data)
		{
			yolo_pub.publish(yolo_sign);
			std::cout << "sign change!!" << std::endl;
		}
        prev_yolo_sign.data = yolo_sign.data;
		loop_rate.sleep();
	}
	return 0;
}



	
