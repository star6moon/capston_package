#ifndef __UTILS_H__
#define __UTILS_H__

#include <thread>
#include "windowbox.h"

template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{
	std::vector<double> linspaced;
	double start = static_cast<double>(start_in);
	double end = static_cast<double>(end_in);
	double num = static_cast<double>(num_in);

	if (num == 0) { return linspaced; }
	if (num == 1) {
		linspaced.push_back(start);
		return linspaced;
	}

	double delta = (end - start) / (num - 1);

	for (int i = 0; i < num - 1; ++i) {
		linspaced.push_back(start + delta * i);
	}
	linspaced.push_back(end);

	return linspaced;
}

void polyfit(const cv::Mat& src_x, const cv::Mat& src_y, cv::Mat& dst, int order)
{
	CV_Assert((src_x.rows > 0) && (src_y.rows > 0) && (src_x.cols == 1) && (src_y.cols == 1)
		&& (dst.cols == 1) && (dst.rows == (order + 1)) && (order >= 1));
	cv::Mat X;
	X = cv::Mat::zeros(src_x.rows, order + 1, CV_32FC1);
	cv::Mat copy;
	for (int i = 0; i <= order; i++)
	{
		copy = src_x.clone();
		pow(copy, i, copy);
		cv::Mat M1 = X.col(i);
		copy.col(0).copyTo(M1);
	}
	cv::Mat X_t, X_inv;
	transpose(X, X_t);
	cv::Mat temp = X_t * X;
	cv::Mat temp2;
	invert(temp, temp2);
	cv::Mat temp3 = temp2 * X_t;
	cv::Mat W = temp3 * src_y;
	W.copyTo(dst);
}

void calc_warp_points(const cv::Mat& img,
	std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst,
	int& y_bottom, int& y_top, int& imd_offset, int& point_check, int offset = 270)
{
	int nX, nY;
	nX = img.cols;
	nY = img.rows;

	// calculate the vertices of the Region of Interest
	src.push_back(cv::Point2f(img.cols/2 - imd_offset, y_top));
	src.push_back(cv::Point2f(img.cols/2 + imd_offset, y_top));
	src.push_back(cv::Point2f(img.cols, y_bottom));
	src.push_back(cv::Point2f(0, y_bottom));

	// calculate the destination points of the warp
	dst.push_back(cv::Point2f(offset, 0));
	dst.push_back(cv::Point2f(nX - offset, 0));
	dst.push_back(cv::Point2f(nX - offset, nY));
	dst.push_back(cv::Point2f(offset, nY));
	//std::cout << "nX : " << nX << "   nY : " << nY << "   dst point nX - offset :" << nX - offset << std::endl;

	if (point_check == 1){
		cv::Mat point_mark = img.clone();
		cv::Scalar orange(0, 165, 255), blue(255, 0, 0), magneta(255, 0, 255), red(0, 0, 255);
		cv::Scalar white(255, 255, 255), black(0, 0, 0);
		cv::circle(point_mark, src[0], 10, orange, 3);
		cv::circle(point_mark, src[1], 10, blue, 3);
		cv::circle(point_mark, src[2], 10, magneta, 3);
		cv::circle(point_mark, src[3], 10, red, 3);
		cv::imshow("point_mark", point_mark);
	}

	return;
}

inline void perspective_warp(const cv::Mat& img, cv::Mat& dst, const cv::Mat& M)
{
	cv::warpPerspective(img, dst, M, img.size(), cv::INTER_LINEAR);
	return;
}

inline void perspective_transforms(std::vector<cv::Point2f> const& src, std::vector<cv::Point2f>  const& dst,
	cv::Mat& M, cv::Mat& Minv)
{
	M = cv::getPerspectiveTransform(src, dst);
	Minv = cv::getPerspectiveTransform(dst, src);

	return;
}


void abs_sobel_thresh(cv::Mat const& src, cv::Mat& dest, char orient = 'x', int kernel_size = 3, int thresh_min = 0, int thresh_max = 255)
{
	int dx, dy;
	int ddepth = CV_64F;

	cv::Mat grad_img, scaled;

	if (orient == 'x') {
		dy = 0;
		dx = 1;
	}
	else {
		dy = 1;
		dx = 0;
	}

	cv::Sobel(src, grad_img, ddepth, dx, dy, kernel_size);
	grad_img = cv::abs(grad_img);

	// Scaling 
	double min, max;
	cv::minMaxLoc(grad_img, &min, &max);
	scaled = 255 * (grad_img / max);
	scaled.convertTo(scaled, CV_8UC1);

	assert(scaled.type() == CV_8UC1);
	cv::inRange(scaled, cv::Scalar(thresh_min), cv::Scalar(thresh_max), dest);

	return;
}


void combined_threshold(cv::Mat const& img, cv::Mat& dst)
{
	// convert to HLS color space
	cv::Mat undist_hls;
	cv::cvtColor(img, undist_hls, cv::COLOR_BGR2HLS);

	// split into H,L,S channels
	cv::Mat hls_channels[3];
	cv::split(undist_hls, hls_channels);

	// apply Absolute Sobel Threshold
	cv::Mat sobel_x, sobel_y, combined;

	/*abs_sobel_thresh(hls_channels[2], sobel_x, 'x', 3, 10, 170);
	abs_sobel_thresh(hls_channels[2], sobel_y, 'y', 3, 10, 170);*/

	// perform thresholding on parallel
        int kernel = 3;
	int tre_max = 255;
	int tre_min = 10;
	std::thread x_direct(abs_sobel_thresh, std::ref(hls_channels[1]), std::ref(sobel_x), 'x', kernel, tre_min, tre_max);
	std::thread y_direct(abs_sobel_thresh, std::ref(hls_channels[1]), std::ref(sobel_y), 'y', kernel, tre_min, tre_max);
	x_direct.join();
	y_direct.join();

	dst = sobel_x & sobel_y; // combine gradient images

	return;
}


void binary_topdown(const cv::Mat& undistorted, cv::Mat& warped,cv::Mat& M,cv::Mat& Minv, int& y_bottom, int& y_top, int& mid_offset, int& point_check)
{

	// top down view warp of the undistorted binary image
	// int y_bottom = undistorted.rows;
	// int y_top = undistorted.rows/2;
	std::vector<cv::Point2f> src, dst;
	calc_warp_points(undistorted, src, dst, y_bottom, y_top, mid_offset, point_check, undistorted.cols * 0.42);

	// calculate matrix for perspective warp
	perspective_transforms(src, dst, M, Minv);
	cv::Mat warped2;
	perspective_warp(undistorted, warped2, M);
	cv::imshow("warped_bgr",warped2);

	// TODO: handle daytime shadow images
	// convert to HLS color space
	cv::Mat combined;
	combined_threshold(undistorted, combined);

	// get a warped image
	perspective_warp(combined, warped, M);
	//cv::imshow("warped",warped);
}


inline void lane_histogram(cv::Mat const& img, cv::Mat& histogram)
{
	// Histogram 
	cv::Mat cropped = img(cv::Rect(0, img.rows / 4, img.cols, img.rows / 4 * 3));
	cv::reduce(cropped / 255, histogram, 0, cv::REDUCE_SUM, CV_32S);

	return;
}

void lane_peaks(cv::Mat const& histogram, cv::Point& left_max_loc, cv::Point& right_max_loc)
{
	// TODO: find a method to handle shadows
	cv::Point temp;
	double min, max;
	int midpoint = histogram.cols / 2;

	cv::Mat left_half = histogram.colRange(0, midpoint - int(midpoint * 0.25));
	cv::Mat right_half = histogram.colRange(midpoint + int(midpoint * 0.25), histogram.cols);
	//std::cout << "right_half : " << right_half << std::endl;

	cv::minMaxLoc(left_half, &min, &max, &temp, &left_max_loc);
	cv::minMaxLoc(right_half, &min, &max, &temp, &right_max_loc);
	right_max_loc = right_max_loc + cv::Point(midpoint + int(midpoint * 0.25), 0);

	return;
}

void find_lane_windows(cv::Mat& binary_img, WindowBox& window_box, std::vector<WindowBox>& wboxes)
{
	bool continue_lane_search = true;
	int contiguous_box_no_line_count = 0;

	int loop_count = 0;

    // std::cout << " loop part find windows start" << std::endl;
	// keep searching up the image for a lane lineand append the boxes
	while (continue_lane_search && window_box.y_top > 0) {
		loop_count += 1;
        // std::cout << "   loop part find 2 !" << std::endl;
        // std::cout << "   window_box.y_top : " << window_box.y_top << std::endl;
        // std::cout << "   continue_lane_search : " << continue_lane_search << std::endl;
        // std::cout << "   window_box.has_line() : " << window_box.has_line() << std::endl;
		// std::cout << "   window_box.has_lane() : " << window_box.has_lane() << std::endl;
		// std::cout << "   contiguous_box_no_line_count : " << contiguous_box_no_line_count << std::endl;
		if (window_box.has_line())
			wboxes.push_back(window_box);
			// std::cout << "   push_box!! loop : " << loop_count << std::endl;
		window_box = window_box.get_next_windowbox(binary_img);

		// if we've found the lane and can no longer find a box with a line in it
		// then its no longer worth while searching
		if (window_box.has_lane())
			if (window_box.has_line())
				contiguous_box_no_line_count = 0;
			else {
				contiguous_box_no_line_count += 1;
				if (contiguous_box_no_line_count >= 4)
					continue_lane_search = false;
					// std::cout << " stop search" << std::endl;
			}
	}
    // std::cout << "-loop part find windows end   wboxes.size() : " << wboxes.size() << std::endl;
	return;
}

void poly_fitx(std::vector<double> const& fity, std::vector<double>& fitx, cv::Mat const& line_fit)
{
	for (auto const& y : fity) {
		double x = line_fit.at<float>(2, 0) * y * y + line_fit.at<float>(1, 0) * y + line_fit.at<float>(0, 0);
		fitx.push_back(x);
	}

	return;
}

void poly_fitx_plus(std::vector<double> const& fity, std::vector<double>& fitx, cv::Mat const& line_fit, float& angle)
{
	for (auto const& y : fity) {
		double x = line_fit.at<float>(2, 0) * y * y + line_fit.at<float>(1, 0) * y + line_fit.at<float>(0, 0);
		fitx.push_back(x);
	}

	angle = 2 * line_fit.at<float>(2, 0) * fity.back() + line_fit.at<float>(1, 0);
	return;
}

void calc_lane_windows(cv::Mat& binary_img, int nwindows, int width,
	std::vector<WindowBox>& left_boxes, std::vector<WindowBox>& right_boxes)
{
	// calc height of each window
	int ytop = binary_img.rows;
	int height = ytop / nwindows;

	// find leftand right lane centers to start with
	cv::Mat histogram;
	lane_histogram(binary_img, histogram); // Histogram 

	cv::Point peak_left, peak_right;
	lane_peaks(histogram, peak_left, peak_right); // Peaks
	if(peak_left.x < width / 2){
		peak_left.x = width / 2;
		// std::cout << "--too left : " << width / 2 << std::endl;
	}
	
	if(peak_right.x > binary_img.cols - width / 2){
		peak_right.x = binary_img.cols - width / 2;
		// std::cout << "--too right : " << binary_img.cols - width / 2 << std::endl;	
	}
	peak_right.x = int(binary_img.cols * 0.6);
	// Initialise left and right window boxes
	WindowBox wbl(binary_img, peak_left.x, ytop, width, height);
	WindowBox wbr(binary_img, peak_right.x, ytop, width, height);
	/*find_lane_windows(binary_img, wbl, left_boxes);
	find_lane_windows(binary_img, wbr, right_boxes);*/
	// Parallelize searching

	std::thread left(find_lane_windows, std::ref(binary_img), std::ref(wbl), std::ref(left_boxes));
	std::thread right(find_lane_windows, std::ref(binary_img), std::ref(wbr), std::ref(right_boxes));
	left.join();
	right.join();

	return;
}

cv::Mat calc_fit_from_boxes(std::vector<WindowBox> const& boxes)
{
	int n = boxes.size();
	std::vector<cv::Mat> xmatrices, ymatrices;
	xmatrices.reserve(n);
	ymatrices.reserve(n);

	// std::cout << " boxes.size() : " << n << std::endl;

	cv::Mat xtemp, ytemp;
	for (auto const& box : boxes) {
		// get matpoints
		box.get_indices(xtemp, ytemp);
		xmatrices.push_back(xtemp);
		ymatrices.push_back(ytemp);
	}
	cv::Mat xs, ys;
	cv::vconcat(xmatrices, xs);
	cv::vconcat(ymatrices, ys);

    // std::cout << " loop part calcfit 1 !" << std::endl;

	// std::cout << " ys.size() : " << ys.size() << std::endl;
	// std::cout << " xs.size() : " << xs.size() << std::endl;

	// Fit a second order polynomial to each
	cv::Mat fit = cv::Mat::zeros(3, 1, CV_32F);
	polyfit(ys, xs, fit, 2);
    

	return fit;
}

cv::Mat calc_fit_from_prev_fit(const cv::Mat& src, cv::Mat& dst, std::vector<double>& fity, std::vector<double>& right_fitx)
{
	std::vector<std::vector<cv::Point>> fit_polygon;
	fit_polygon.push_back(std::vector<cv::Point>());

	int margin = src.cols/11;
	for(int i = 0; i < fity.size(); i++ ){
		if(right_fitx[i] - margin < 0){
			fit_polygon[0].push_back(cv::Point(0, fity[i]));
		}
		else{
			fit_polygon[0].push_back(cv::Point(right_fitx[i] - margin, fity[i]));
		}
		
	}
	for(int i = fity.size() - 1; i >= 0; i--){
		if(right_fitx[i] + margin > src.cols - 1){
			fit_polygon[0].push_back(cv::Point(src.cols - 1, fity[i]));
		}
		else{
			fit_polygon[0].push_back(cv::Point(right_fitx[i] + margin, fity[i]));
		}
	}
	cv::Mat mask(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
	drawContours(mask, fit_polygon,0, cv::Scalar(255), cv::FILLED, 8 );
	src.copyTo(dst, mask);

	std::vector<cv::Point> nonzero;
	cv::findNonZero(dst, nonzero);
	if(nonzero.size() <= 10){
		cv::Mat fit = cv::Mat::zeros(3, 1, CV_32F);
		return fit;
	}

	int npoints = nonzero.size();
	cv::Mat x = cv::Mat::zeros(npoints, 1, CV_32F);
	cv::Mat y = cv::Mat::zeros(npoints, 1, CV_32F);

	for (int i = 0; i < npoints; i++) {
		x.at<float>(i, 0) = nonzero[i].x;
		y.at<float>(i, 0) = nonzero[i].y;
	}

	cv::Mat fit = cv::Mat::zeros(3, 1, CV_32F);
	polyfit(y, x, fit, 2);
	return fit;
}

void draw_line(cv::Mat& img, cv::Mat& left_fit, cv::Mat& right_fit, cv::Mat Minv, cv::Mat& out_img)
{
	int y_max = img.rows;
	std::vector<double> fity = linspace<double>(0, y_max - 1, y_max);
	cv::Mat color_warp = cv::Mat::zeros(img.size(), CV_8UC3);

	// Calculate Points
	std::vector<double> left_fitx, right_fitx;
	poly_fitx(fity, left_fitx, left_fit);
	poly_fitx(fity, right_fitx, right_fit);

	int npoints = fity.size();
	std::vector<cv::Point> pts_left(npoints), pts_right(npoints), pts;
	for (int i = 0; i < npoints; i++) {
		pts_left[i] = cv::Point(left_fitx[i], fity[i]);
		pts_right[i] = cv::Point(right_fitx[i], fity[i]);
	}
	pts.reserve(2 * npoints);
	pts.insert(pts.end(), pts_left.begin(), pts_left.end());
	pts.insert(pts.end(), pts_right.rbegin(), pts_right.rend());
	std::vector<std::vector<cv::Point>> ptsarray{ pts };
	cv::fillPoly(color_warp, ptsarray, cv::Scalar(0, 255, 0));

	cv::Mat new_warp;
	perspective_warp(color_warp, new_warp, Minv);
	cv::addWeighted(img, 1, new_warp, 0.3, 0, out_img);

	return;
}

void draw_boxes(cv::Mat& img, const std::vector<WindowBox>& boxes)
{
	// Draw the windows on the output image
	cv::Point pnt1, pnt2;
	for (const auto& box : boxes) {
		pnt1 = box.get_bottom_left_point();
		pnt2 = box.get_top_right_point();
		cv::rectangle(img, pnt1, pnt2, cv::Scalar(0, 255, 0), 2);
	}

	return;
}

void draw_polyline(cv::Mat& out_img, std::vector<double> const& fitx, std::vector<double> const& fity, cv::Scalar& color)
{
	assert(fitx.size() == fity.size());

	std::vector<cv::Point2f> points;
	for (int i = 0; i < fity.size(); i++)
		points.push_back(cv::Point2f(fitx[i], fity[i]));
	cv::Mat curve(points, true);
	curve.convertTo(curve, CV_32S); //adapt type for polylines
	cv::polylines(out_img, curve, false, color, 2);

}

void place_img(cv::Mat& src, cv::Mat& dst, cv::Rect& roi, cv::Point& piece)
{
	cv::Mat small_img;
	if (src.channels() != 3) {
		auto channels = std::vector<cv::Mat>{ src,src,src };
		cv::merge(channels, small_img);
		cv::resize(small_img, small_img, cv::Size(piece.x, piece.y), 0, 0, cv::INTER_LINEAR);
		small_img.copyTo(dst(roi));
	}
	else {
		cv::resize(src, src, cv::Size(piece.x, piece.y), 0, 0, cv::INTER_LINEAR);
		src.copyTo(dst(roi));
	}
	return;
}








#endif
