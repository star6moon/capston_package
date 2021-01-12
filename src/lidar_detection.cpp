#include "ros/ros.h"
#include <sensor_msgs/LaserScan.h>
#include <std_msgs/Float64.h>
#include <cmath>
#include <algorithm>
#include <vector>
using namespace std;

#define _USE_MATH_DEFINES

vector<float> current_ranges;
static int angle1 = 0;
static int angle2 = 359;
static int index1 = 0;
static int index2 = 0;
static int rchange = 0;
static float min_distance;
static int stop = 0;
static int first = 0;
static int abc = 0;
static int direction = 0;
static std_msgs::Float64 distance_msg;

void laser_scan_Callback(const sensor_msgs::LaserScan::ConstPtr& msg)
{

	sensor_msgs::LaserScan lidar_val;
	lidar_val.angle_increment = msg->angle_increment;
	lidar_val.ranges = msg->ranges;
	lidar_val.angle_max = msg->angle_max;
	lidar_val.angle_min = msg->angle_min;
	float full_angle = lidar_val.angle_max - lidar_val.angle_min;
	float len = full_angle/lidar_val.angle_increment;

	index1 = angle1/180.0*M_PI/lidar_val.angle_increment;
	index2 = angle2/180.0*M_PI/lidar_val.angle_increment;
	min_distance = 5;

	if(rchange == 1)
	{
		current_ranges.resize(index2-index1);
	}
	for(int i = 0; i < index2 - index1; i++)
	{
		current_ranges.push_back(lidar_val.ranges[i+index1]);
		if((lidar_val.ranges[i+index1] > 0)&(lidar_val.ranges[i+index1] < min_distance))
		{	
			min_distance = lidar_val.ranges[i+index1];
			direction = i+index1;
		}
	}

	//min_distance = *min_element(current_ranges.begin(), current_ranges.end());
	if((min_distance > 0)&(min_distance < 0.30)){stop = 1;}
	else{stop = 0;}

	if(first == 0)
	{
		printf("full_angle = %f , len = %f \n",full_angle/M_PI*180,len);
		printf("ranges[%f] = %f \n",len, lidar_val.ranges[len]);	
		first = 1;
	}

	if(abc % 4 == 0)
	{	
		printf("min_distance  = %f  at %d\n",min_distance, direction);
		abc = 0;
	}

	distance_msg.data = lidar_val.ranges[0];
	++abc;

}


int main(int argc, char **argv)
{

	ros::init(argc, argv, "lidar_detection");
	ros::NodeHandle nh;
	ros::Publisher lidar_distance_pub = nh.advertise<std_msgs::Float64>("/cognition/obj_distance", 1);
	ros::Subscriber lidar_sub = nh.subscribe("/scan", 1, laser_scan_Callback);
	while(ros::ok())
	{	
		ros::Rate loop_rate(4);
		ros::spinOnce();
		lidar_distance_pub.publish(distance_msg);
		loop_rate.sleep();
	}

	return 0;
}


