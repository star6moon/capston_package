#include "opencv2/opencv.hpp"
#include "utils.h"
#include "windowbox.h"
#include <iostream>
#include "ros/ros.h"
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Float64.h>
//#include <cv_bridge/cv_bridge.h>
//#include <image_transport/image_transport.h>

// using namespace cv;
// using namespace std;


// Window parameters
#define N_WINDOWS 10
#define WINDOW_WIDTH 50

double y_top_ratio, mid_offset_ratio;
std::string device_num;
double peak_l, peak_r1, peak_r2;
double check_val, check_img;
double mode;
double lane_offset;
void state_Callback(const std_msgs::Float64::ConstPtr& msg)
{
    mode = msg->data;
}

int main(int argc, char **argv){
	ros::init(argc, argv, "lane_detection");
	ros::NodeHandle nh;
	ros::NodeHandle private_nh("~");
	ros::Publisher lane_publisher = nh.advertise<std_msgs::Float64MultiArray>("/cognition/lane_condition", 1);
	ros::Subscriber midterm_imu_sub = nh.subscribe("/state/lane_switch", 1, state_Callback);
	ros::Rate loop_rate(3);
	private_nh.param("device_num", device_num, std::string("/dev/video0"));
	private_nh.param("y_top_ratio", y_top_ratio, 0.4);
	private_nh.param("mid_offset_ratio", mid_offset_ratio, 0.125);
	private_nh.param("peak_l", peak_l, 0.4);
	private_nh.param("peak_r1", peak_r1, 0.6);
	private_nh.param("peak_r2", peak_r2, 0.8);
	private_nh.param("check_val", check_val, 0.0);
	private_nh.param("check_img", check_img, 0.0);
	private_nh.param("mode", mode, 1.0);
	private_nh.param("lane_offset", lane_offset, 100.0);
	//ros::Publisher img_publisher = nh.advertise<sensor_msgs::Image>("/cognition/image", 1);
    //웹캡으로 부터 데이터 읽어오기 위해 준비
    // 보통 0 또는 1번입니다
    // 노트북의 경우 웹캠이 자체적으로 달려있으므로 1번일 가능성이 높습니다
    cv::VideoCapture cap1("/dev/video0");
	// cv::VideoCapture cap1("tcambin serial = 50910677! video / x-raw, format = BGRx, width = 1280, height = 960, framerate = 20 / 1! videoconvert! videoscale! appsink", cv::CAP_GSTREAMER);
    // cv::VideoCapture cap1(0);


	int f_width = cap1.get(cv::CAP_PROP_FRAME_WIDTH);
	int f_height = cap1.get(cv::CAP_PROP_FRAME_HEIGHT);
	int f_fps = cap1.get(cv::CAP_PROP_FPS);
	std::cout << "initial width : " << f_width << "  height : " << f_height << "  fps : " << f_fps << std::endl;

	// cap1.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
	cap1.set(cv::CAP_PROP_FPS, 5);
	//cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	//cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	cap1.set(cv::CAP_PROP_FRAME_WIDTH, 480);
	cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 320);
	// cap1.set(cv::CAP_PROP_FRAME_WIDTH, f_width/2);
	// cap1.set(cv::CAP_PROP_FRAME_HEIGHT, f_height/2);

	f_width = cap1.get(cv::CAP_PROP_FRAME_WIDTH);
	f_height = cap1.get(cv::CAP_PROP_FRAME_HEIGHT);
	f_fps = cap1.get(cv::CAP_PROP_FPS);
	std::cout << "after set width : " << f_width << "  height : " << f_height << "  fps : " << f_fps << std::endl;

	if (!cap1.isOpened()) 
        std::cout << "첫번째 카메라를 열 수 없습니다." << std::endl;

    cv::Mat frame, warped, Minv, M;
    cap1 >> frame;
	
    int nframe = 0;
    int height = frame.rows;
	int restart = 1;
	cv::Point piece(frame.cols/4, frame.rows/4);
	cv::Scalar red(0, 0, 255), blue(255, 0, 0), yellow(153, 255, 255);
	cv::Scalar orange(0, 165, 255), magneta(255, 0, 255);
	cv::Scalar white(255, 255, 255), black(0, 0, 0);
	cv::Rect r1(cv::Point(piece.x * 0, 0), cv::Point(piece.x * 1, piece.y));
	cv::Rect r2(cv::Point(piece.x * 1, 0), cv::Point(piece.x * 2, piece.y));
	cv::Rect r3(cv::Point(piece.x * 2, 0), cv::Point(piece.x * 3, piece.y));
	cv::Rect r4(cv::Point(piece.x * 3, 0), cv::Point(piece.x * 4, piece.y));

	std_msgs::Float64MultiArray msg;
	msg.data.resize(6);

	std::vector<WindowBox> left_boxes, right_boxes;
	std::vector<double> fitx, fity, left_fitx, right_fitx, hist_fity, new_left_fitx, new_right_fitx;
	float L_angle, R_angle, L_dist, R_dist;
	L_angle = R_angle = L_dist = R_dist = 0;
	cv::Mat left_fit, right_fit;

    std::cout << "loop start !" << std::endl;



    while (1) {
		static int fail_L = 1;
		static int fail_R = 1;

		//웹캡으로부터 한 프레임을 읽어옴
        cap1 >> frame;
        // q키를 누르면 종료

        if (cv::waitKey(1) == 113 || frame.empty()){
			break;
		}
        nframe++;

		ros::spinOnce();
		if(mode == 0){
			fail_L = 1;
			fail_R = 1;
			std::cout << "mode : 0   -   wait for changing mode" << std::endl;
			loop_rate.sleep();
			continue;
		}


        // system("cls"); // window clear
		static int y_bottom = frame.rows;
		static int y_top = frame.rows * y_top_ratio;
		static int mid_offset = frame.cols * mid_offset_ratio;
		static int point_check = 1;
        binary_topdown(frame, warped, M, Minv, y_bottom, y_top, mid_offset, point_check);

		// generate x and values for plotting
		fity = linspace<double>(0, warped.rows - 1, warped.rows);
		fitx = linspace<double>(0, warped.cols - 1, warped.cols);

		cv::Mat histogram;
		lane_histogram(warped, histogram);
		hist_fity.clear();
		for (int i = 0; i < histogram.cols; i++)
			hist_fity.push_back(height - histogram.at<int>(0, i));

		if(mode >= 1){
			if(fail_R){
				left_boxes.clear();
				right_boxes.clear();
				if(mode == 3){
					calc_lane_windows(warped, N_WINDOWS, WINDOW_WIDTH, left_boxes, right_boxes, peak_l, peak_r2, 1); // 0 : left,  1 : right,  2 : both
				}
				else{
					calc_lane_windows(warped, N_WINDOWS, WINDOW_WIDTH, left_boxes, right_boxes, peak_l, peak_r1, 1); // 0 : left,  1 : right,  2 : both
				}

				int min_box_num = 6;
				if(right_boxes.size() >= min_box_num){
					right_fit = calc_fit_from_boxes(right_boxes);
					right_fitx.clear();
					poly_fitx_plus(fity, right_fitx, right_fit, R_angle);
					fail_R = 0;
				}
				else{
					fail_R = 1;
					std::cout << "fail_R  : 1.find window fail !!   -   right_boxes.size() : "<< right_boxes.size() << std::endl;
				}
			}
			if(!fail_R){
				cv::Mat masked;
				right_fit = calc_fit_from_prev_fit(warped, masked, fity, right_fitx);
				cv::imshow("masked", masked);
				if(right_fit.at<float>(2, 0) == 0){
					fail_R = 1;
					std::cout << "fail_R  : 2.find fit fail !!" << std::endl;
				}
				right_fitx.clear();
				poly_fitx_plus(fity, right_fitx, right_fit, R_angle);
				R_dist = right_fitx.back() - warped.cols/2;
				fail_R = 0;
				// std::cout << "Debug - right_fitx.back() - warped.cols/2 : "<< right_fitx.back() - warped.cols/2 << std::endl;
				
				if((R_dist < 0) || (R_dist > warped.cols/2)){
					fail_R = 1;
					std::cout << "fail_R  : 3.fit lane is weird !!" << std::endl;
				} 
			}
		}
		if (mode == 2){
			if(fail_L){
				left_boxes.clear();
				right_boxes.clear();
				// std::cout << "fail_R : 3.fit lane is weird !!" << std::endl;
				calc_lane_windows(warped, N_WINDOWS, WINDOW_WIDTH, left_boxes, right_boxes, peak_l, peak_r1, 0); // 0 : left,  1 : right,  2 : both

				int min_box_num = 6;
				if(left_boxes.size() >= min_box_num){
					left_fit = calc_fit_from_boxes(left_boxes);
					left_fitx.clear();
					poly_fitx_plus(fity, left_fitx, left_fit, R_angle);
					fail_L = 0;
				}
				else{
					fail_L = 1;
					std::cout << "fail_L  : 1.find window fail !!   -   left_boxes.size() : "<< left_boxes.size() << std::endl;
				}
			}
			if(!fail_L){
				cv::Mat masked;
				left_fit = calc_fit_from_prev_fit(warped, masked, fity, left_fitx);
				cv::imshow("masked", masked);
				if(left_fit.at<float>(2, 0) == 0){
					fail_L = 1;
					std::cout << "fail_L  : 2.find fit fail !! !!" << std::endl;
				}
				left_fitx.clear();
				poly_fitx_plus(fity, left_fitx, left_fit, L_angle);
				L_dist = warped.cols/2 - left_fitx.back();
				fail_L = 0;
				
				if((L_dist < 0) || (L_dist > warped.cols/2)){
					fail_L = 1;
					std::cout << "fail_L  : 3.fit lane is weird !!   -   L_dist : " << L_dist << std::endl;
				} 
				if(L_dist + R_dist < lane_offset){
					std::cout << "fail_LR : 4.fit lane is weird !!   -   L_dist + R_dist : " << L_dist + R_dist << std::endl;
					fail_L = 1;
					fail_R = 1;
				}
			}
		}
		if(mode == 4){
			color_detection(frame);
		}

		if(check_val == 1){
			std::cout << "L_angle : " << L_angle << "   L_dist : " << L_dist << "   R_angle : " << R_angle << "   R_dist : " << R_dist << std::endl;
		}


		cv::Mat warp_back;
		if(!fail_L || !fail_R){
			warp_back = cv::Mat::zeros(frame.size(), CV_8UC3);
			draw_line(frame, left_fit, right_fit, Minv, warp_back, fail_L, fail_R);
		}
		// std::cout << "--5" << std::endl;

		cv::Mat out_img;
		auto channels = std::vector<cv::Mat>{ warped,warped,warped };
		cv::merge(channels, out_img);
		//std::cout << "--6" << std::endl;

		if(check_img == 1){
			if(!fail_L){
				draw_polyline(out_img, left_fitx, fity, blue);
			}
			else{
				draw_boxes(out_img, left_boxes);
			}
			if(!fail_R){
				draw_polyline(out_img, right_fitx, fity, blue);
			}
			else{
				draw_boxes(out_img, right_boxes);				
			}
			// draw_polyline(out_img, L_fitx1, fity, red);
			// cv::circle(out_img, cv::Point2f(L_cross_x, L_cross_y), 3, orange, 3);
			// cv::circle(out_img, cv::Point2f(R_cross_x, R_cross_y), 3, red, 3);
			cv::imshow("outimg", out_img);

			if(!fail_L || !fail_R){
				//hls[1]
				cv::Mat hls[3], dst;
				cv::cvtColor(frame, dst, cv::COLOR_BGR2HLS);
				cv::split(dst, hls);
	
				place_img(hls[2], warp_back, r1, piece);
	
				//binary  combined
				combined_threshold(frame, dst);
				place_img(dst, warp_back, r2, piece);

				//windows
				place_img(out_img, warp_back, r3, piece);

				//hisotgram
				cv::Mat black_img(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
				draw_polyline(black_img, fitx, hist_fity, red);
				place_img(black_img, warp_back, r4, piece);

				cv::imshow("warp_back1", warp_back);
			}
		}


		msg.data[0] = L_angle;
		msg.data[1] = R_angle;
		msg.data[2] = L_dist;
		msg.data[3] = R_dist;
		msg.data[4] = fail_L;
		msg.data[5] = fail_R;
		lane_publisher.publish(msg);

		//std::cout << "warped.size() :" << warped.size() << "   warped.type() : " << warped.type() << std::endl;
		//sensor_msgs::Image img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", warped).toImageMsg();
		//img_publisher.publish(img_msg);
		// std::cout << "--7" << std::endl;
    }

	// When everything done, release the video capture object
	cap1.release();
	std::cout << "Break!!" << std::endl;

    return 0;
}
