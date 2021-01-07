#include "ros/ros.h"
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>
#include <cmath>

#define _USE_MATH_DEFINES

static float distance = 0;
static int angle_lock = 0;
static int distance_lock = 0;
static int yolo_sign = 0;
static int yolo_lock = 0;
static float vel_val = 0;
static double x_roll = 0;
static double y_pitch = 0;
static double z_yaw = 0;
static geometry_msgs::Twist cmd_vel;
static geometry_msgs::Twist prev_vel;

static float L_angle = 0; 
static float R_angle = 0;
static float L_dist = 0;
static float R_dist = 0;
static float left_or_right = 0;
static float th_target = 0;
static float th_max = 0;
static float th_err = 0;
static float prev_th_err = 0;
static float integ_th_err = 0;
static float gain = 0;
static float vel_max = 0;
static float vel_min = 0;
static float kp = 0;
static float ki = 0;
static float kd = 0;

void obj_Callback(const std_msgs::Float64::ConstPtr& msg)
{
	distance = msg->data;
	if((distance>0)&(distance<0.3))
	{
		distance_lock = 1;
		printf("distance_lock!! : %f\n",distance);
	}
	else
	{
		distance_lock = 0;
	}
}

void yolo_Callback(const std_msgs::Float64::ConstPtr& msg)
{
    yolo_sign = msg->data;
	if(yolo_sign == 1){
		yolo_lock = 1;
		std::cout << "yolo_lock!!" << std::endl;
	}
	else{
		yolo_lock = 0;
	}
}

void lane_Callback(const std_msgs::Float64MultiArray::ConstPtr& msg)
{
    L_angle = msg->data[0]; 
    R_angle = msg->data[1];
    L_dist = msg->data[2] / 1000 * 2;
    R_dist = msg->data[3] / 1000 * 2;


	float Len_target = 0.3;
	float lane_width = 0.28;

	// angle : left(+), right(-). (if car is on the rightside, th_target is +)
	left_or_right = 0;
	if(left_or_right == 0){
		th_target = atan2(L_dist - lane_width, Len_target);
		th_err = th_target + L_angle;
	}
	else{
		th_target = atan2(lane_width - R_dist, Len_target);
		th_err = th_target + R_angle;
	}

	double pi = M_PI;
	th_max = 45.0 / 180.0 * pi;

	if(th_target > th_max){
		th_target = th_max;
	}
	else if(th_target < -th_max){
		th_target = -th_max;
	}


	vel_max = 3;
	vel_min = 0.2;


	integ_th_err += th_err;
	kp = 1000;
	kd = 1;
	ki = 0.0001;

	gain = kp * th_err + ki * (integ_th_err) + kd * (prev_th_err - th_err);

    vel_val = vel_min + (vel_max - vel_min) * (th_max - std::abs(th_target)) / th_max;
    cmd_vel.angular.z = gain / 1000;

	prev_th_err = th_err;

	std::cout << "L_dist - lane_width : " << L_dist - lane_width << "   lane_width - R_dist : " << lane_width - R_dist << std::endl;	
	std::cout << "th_target : " << th_target << "   th_err : " << th_err << "   gain : " << gain << "   vel_val : " << vel_val << std::endl;
}

void imu_Callback(const geometry_msgs::Vector3::ConstPtr& msg)
{
	x_roll = msg->x * 180/M_PI;
	y_pitch = msg->y * 180/M_PI;
	z_yaw = msg->z * 180/M_PI;

	if(((x_roll>45)|(x_roll<-45))|((y_pitch>45)|(y_pitch<-45)))
	{
		angle_lock = 1;
		printf("angle_lock!!\n");
	}
	else
	{
		angle_lock = 0;
	}
}

int main(int argc, char **argv)
{

	ros::init(argc, argv, "planning_node");
	ros::NodeHandle nh;
	ros::Subscriber lane_sub = nh.subscribe("/cognition/lane_condition", 1, lane_Callback);
    ros::Subscriber yolo_sub = nh.subscribe("/cognition/yolo", 1, yolo_Callback);
    ros::Subscriber obj_sub = nh.subscribe("/cognition/obj_distance", 1, obj_Callback);
    ros::Subscriber imu_sub = nh.subscribe("/cognition/euler_angle", 1, imu_Callback);
	ros::Publisher final_control_pub = nh.advertise<geometry_msgs::Twist>("/control/cmd_vel", 1);
	
	while(ros::ok())
	{	
		ros::Rate loop_rate(7);
		ros::spinOnce();

        if((yolo_lock==0)&(angle_lock==0)&(distance_lock==0)){
			cmd_vel.linear.x = vel_val;
		}
		else{
			cmd_vel.linear.x = 0;
		}

		if((prev_vel.linear.x != cmd_vel.linear.x)||(prev_vel.angular.z != cmd_vel.angular.z)){
			printf("vel_pub!\n");
			final_control_pub.publish(cmd_vel);
		}

        prev_vel = cmd_vel;		
		loop_rate.sleep();

	}	
	
	return 0;
}