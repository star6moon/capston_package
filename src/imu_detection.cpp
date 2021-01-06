#include "ros/ros.h"
#include "sensor_msgs/Imu.h"
#include "geometry_msgs/Vector3.h"
#include <cmath>
#include <algorithm>
using namespace std;
struct EulerAngles {
    double x_roll, y_pitch, z_yaw;
};
struct Quaternion
{
    double w, x, y, z;
};
static sensor_msgs::Imu imu;
static geometry_msgs::Vector3 Euler_angles; 
static EulerAngles angles1;
static Quaternion q1;

EulerAngles Quaternion2Euler(Quaternion q)
{
	EulerAngles angles;
	float X = imu.orientation.x;
	float Y = imu.orientation.y;
	float Z = imu.orientation.z;
	float W = imu.orientation.w;

	double t0 = 2 * (W * X + Y * Z);
	double t1 = 1 - 2 * (X * X + Y * Y);
	angles.x_roll = atan2(t0, t1);

	double t2 = 2 * (W * Y - Z * X);
	if (abs(t2) >= 1)
        	angles.y_pitch = copysign(M_PI / 2, t2);
    	else
        	angles.y_pitch = asin(t2);

	double t3 = 2 * (W * Z + X * Y);
    	double t4 = 1 - 2 * (Y * Y + Z * Z);
    	angles.z_yaw = atan2(t3, t4);
	
	return angles;
}
Quaternion Euler2Quaternion(EulerAngles angles)
{
	Quaternion q;
	double cy = cos(angles.z_yaw * 0.5);
    double sy = sin(angles.z_yaw * 0.5);
    double cp = cos(angles.y_pitch * 0.5);
    double sp = sin(angles.y_pitch * 0.5);
    double cr = cos(angles.x_roll * 0.5);
    double sr = sin(angles.x_roll * 0.5);

    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

	return q;
}

void imu_Callback(const sensor_msgs::Imu::ConstPtr& imu_msg)
{
	imu.orientation = imu_msg->orientation;
	q1.x = imu.orientation.x;
	q1.y = imu.orientation.y;
	q1.z = imu.orientation.z;
	q1.w = imu.orientation.w;
	
	angles1 = Quaternion2Euler(q1);

	printf(" x_roll  = %.4f\n", angles1.x_roll*180/M_PI);
	printf(" y_pitch = %.4f\n", angles1.y_pitch*180/M_PI);
	printf(" z_yaw   = %.4f\n", angles1.z_yaw*180/M_PI);

	Euler_angles.x = angles1.x_roll;
	Euler_angles.y = angles1.y_pitch;
	Euler_angles.z = angles1.z_yaw;
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "imu_detection");
	ros::NodeHandle nh;
	
	ros::Subscriber midterm_imu_sub = nh.subscribe("/imu", 1, imu_Callback);
	ros::Publisher midterm_imu_pub = nh.advertise<geometry_msgs::Vector3>("/cognition/euler_angle", 1);
	
	printf("start imu_node\n");
	while(ros::ok())
	{	
		ros::Rate loop_rate(10);
		ros::spinOnce();
		midterm_imu_pub.publish(Euler_angles);
		loop_rate.sleep();
	}

	return 0;
}
