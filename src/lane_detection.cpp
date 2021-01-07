#include "opencv2/opencv.hpp"
#include "utils.h"
#include "windowbox.h"
#include <iostream>
#include "ros/ros.h"
#include <std_msgs/Float64MultiArray.h>

// using namespace cv;
// using namespace std;


// Window parameters
#define N_WINDOWS 10
#define WINDOW_WIDTH 100


int main(int argc, char **argv){
	ros::init(argc, argv, "lane_detection");
	ros::NodeHandle nh;
	ros::Publisher lane_publisher = nh.advertise<std_msgs::Float64MultiArray>("/cognition/lane_condition", 1);

    //웹캡으로 부터 데이터 읽어오기 위해 준비
    // 보통 0 또는 1번입니다
    // 노트북의 경우 웹캠이 자체적으로 달려있으므로 1번일 가능성이 높습니다
    cv::VideoCapture cap1("/dev/video0");
	// cv::VideoCapture cap1("tcambin serial = 50910677! video / x-raw, format = BGRx, width = 1280, height = 960, framerate = 20 / 1! videoconvert! videoscale! appsink", cv::CAP_GSTREAMER);
    // cv::VideoCapture cap1(0);


	int f_width = cap1.get(cv::CAP_PROP_FRAME_WIDTH);
	int f_height = cap1.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "initial width : " << f_width << "  height : " << f_height << std::endl;

	// cap1.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
	cap1.set(cv::CAP_PROP_FPS, 10);
	cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	// cap1.set(cv::CAP_PROP_FRAME_WIDTH, f_width/2);
	// cap1.set(cv::CAP_PROP_FRAME_HEIGHT, f_height/2);

	f_width = cap1.get(cv::CAP_PROP_FRAME_WIDTH);
	f_height = cap1.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "after set width : " << f_width << "  height : " << f_height << std::endl;

	if (!cap1.isOpened()) 
        std::cout << "첫번째 카메라를 열 수 없습니다." << std::endl;

    cv::Mat frame, warped, Minv, M;
    cap1 >> frame;
	
    int nframe = 0;
    int height = frame.rows;
	cv::Point piece(frame.cols/4, frame.rows/4);
	cv::Scalar red(0, 0, 255), blue(255, 0, 0), yellow(153, 255, 255);
	cv::Scalar orange(0, 165, 255), magneta(255, 0, 255);
	cv::Scalar white(255, 255, 255), black(0, 0, 0);
	cv::Rect r1(cv::Point(piece.x * 0, 0), cv::Point(piece.x * 1, piece.y));
	cv::Rect r2(cv::Point(piece.x * 1, 0), cv::Point(piece.x * 2, piece.y));
	cv::Rect r3(cv::Point(piece.x * 2, 0), cv::Point(piece.x * 3, piece.y));
	cv::Rect r4(cv::Point(piece.x * 3, 0), cv::Point(piece.x * 4, piece.y));

	std_msgs::Float64MultiArray msg;
	msg.data.resize(4);

    std::cout << "loop start !" << std::endl;

    while (1) {
        //웹캡으로부터 한 프레임을 읽어옴
        cap1 >> frame;
        // q키를 누르면 종료

        if (cv::waitKey(1) == 113 || frame.empty()){
			break;
		}
        nframe++;
		// std::cout << "--0" << std::endl;

        // system("cls"); // window clear
		int y_bottom = frame.rows;
		int y_top = frame.rows/2;
		int mid_offset = 100;
		int point_check = 1;
        binary_topdown(frame, warped, M, Minv, y_bottom, y_top, mid_offset, point_check);
		cv::imshow("warped",warped);
		// std::cout << "--0-1" << std::endl;
        
        cv::Mat histogram;
		lane_histogram(warped, histogram);
		// std::cout << "--0-2" << std::endl;
		
        // Peaks
		cv::Point leftx_base, rightx_base;
		lane_peaks(histogram, leftx_base, rightx_base);
		// std::cout << "--0-3" << std::endl;

		std::vector<WindowBox> left_boxes, right_boxes;
		calc_lane_windows(warped, N_WINDOWS, WINDOW_WIDTH, left_boxes, right_boxes);
		// std::cout << "--0-4" << std::endl;

		int min_box_num = 6;
		if((left_boxes.size() < min_box_num) || (right_boxes.size() < min_box_num) ){
			std::cout << "--countinue left_boxes.size() : " << left_boxes.size() << "   right_boxes.size() : " << right_boxes.size() << std::endl;
			continue;
		}
		else{
			// std::cout << "--process left_boxes.size() : " << left_boxes.size() << "   right_boxes.size() : " << right_boxes.size() << std::endl;
		}
		// std::cout << "--0-5" << std::endl;
			

		cv::Mat left_fit = calc_fit_from_boxes(left_boxes);
		cv::Mat right_fit = calc_fit_from_boxes(right_boxes);
		// std::cout << "--1" << std::endl;

		// generate x and values for plotting
		std::vector<double> fitx, fity, left_fitx, right_fitx, hist_fity, new_left_fitx, new_right_fitx;
		fity = linspace<double>(0, warped.rows - 1, warped.rows);
		fitx = linspace<double>(0, warped.cols - 1, warped.cols);
		// std::cout << "--2" << std::endl;

		for (int i = 0; i < histogram.cols; i++)
			hist_fity.push_back(height - histogram.at<int>(0, i));

		float L_angle, R_angle, L_dist, R_dist;
		poly_fitx_plus(fity, left_fitx, left_fit, L_angle);
		poly_fitx_plus(fity, right_fitx, right_fit, R_angle);
		// std::cout << "--3" << std::endl;
		
		// double L_a1 = (-1/L_angle);
		// double L_b1 = (warped.cols/2) - L_a1 * warped.rows;
		
		// double L_a2 = left_fit.at<float>(2, 0);
		// double L_b2 = left_fit.at<float>(1, 0);
		// double L_c2 = left_fit.at<float>(0, 0);
		
		// double L_cross_y = (-(L_b2 - L_a1) + sqrt(pow(L_b2 - L_a1, 2) - 4 * L_a2 * (L_c2 - L_b1))) / (2 * L_a1);
		// double L_cross_x = L_a1 * L_cross_x + L_b1;

		// std::cout << "L_cross_y : " << L_cross_y << "   L_cross_x : " << L_cross_x << std::endl;
		// L_dist = sqrt(pow((L_cross_x - warped.cols/2), 2) + pow(L_cross_y - (warped.rows-1), 2));
		// std::vector<double> L_fitx1;
		// for (auto const& y : fity) {
		// 	double x = L_a1 * y + L_b1;
		// 	L_fitx1.push_back(x);
		// }


		// double R_a1 = (-1/R_angle);
		// double R_b1 = (warped.rows-1) - R_a1 * warped.cols/2;
		
		// double R_a2 = right_fit.at<float>(2, 0);
		// double R_b2 = right_fit.at<float>(1, 0);
		// double R_c2 = right_fit.at<float>(0, 0);
		
		// double R_cross_x = (-(R_b2 - R_a1) + sqrt(pow(R_b2 - R_a1, 2) - 4 * R_a2 * (R_c2 - R_b1))) / (2 * R_a1);
		// double R_cross_y = R_a1 * R_cross_x + R_b1;

		// R_dist = sqrt(pow((R_cross_x - warped.cols/2), 2) + pow(R_cross_y - (warped.rows-1), 2));

		L_dist = warped.cols/2 - left_fitx.back();
		R_dist = right_fitx.back() - warped.cols/2;
		std::cout << "L_angle : " << L_angle << "   R_angle : " << R_angle << "   L_dist : " << L_dist << "   R_dist : " << R_dist << std::endl;
		// std::cout << "--4" << std::endl;

		cv::Mat warp_back = cv::Mat::zeros(frame.size(), CV_8UC3);
		draw_line(frame, left_fit, right_fit, Minv, warp_back);
		// std::cout << "--5" << std::endl;

		cv::Mat out_img;
		auto channels = std::vector<cv::Mat>{ warped,warped,warped };
		cv::merge(channels, out_img);
		// std::cout << "--6" << std::endl;

		int check_process = 1;
		if(check_process == 1){
			draw_boxes(out_img, left_boxes);
			draw_boxes(out_img, right_boxes);
			draw_polyline(out_img, left_fitx, fity, blue);
			draw_polyline(out_img, right_fitx, fity, blue);
			// draw_polyline(out_img, L_fitx1, fity, red);
			// cv::circle(out_img, cv::Point2f(L_cross_x, L_cross_y), 3, orange, 3);
			// cv::circle(out_img, cv::Point2f(R_cross_x, R_cross_y), 3, red, 3);
			cv::imshow("outimg", out_img);

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


		msg.data[0] = L_angle;
		msg.data[1] = R_angle;
		msg.data[2] = L_dist;
		msg.data[3] = R_dist;
		lane_publisher.publish(msg);
		// std::cout << "--7" << std::endl;
    }

	// When everything done, release the video capture object
	std::cout << "Break!!" << std::endl;
	cap1.release();


    return 0;
}
