#ifndef VIDEO_PROCESSING_LIB_HPP
#define VIDEO_PROCESSING_LIB_HPP

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

using namespace std;
using namespace cv;

void perform_image_preprocessing(Mat& input_image, Mat& gray_image, Mat& binary_image);

double calculate_mean(Mat& features_array);
int perform_convolution(Mat srcConvImage, string kernelFileName, string kernelPattern);

/// FEATURES
void feature_calculation(Mat& grayImage, Mat& binaryImage, ofstream& TextFile, bool save_to_file, Mat &feature_vector);
int find_contour_features(Mat &input_image, int &perimeter, double &area,
		   vector<vector<Point> > &contours);				    // 1
int get_x_y_bar(Mat &binaryImage, double &x_bar, double &y_bar, int area);	    // 2
int get_theta(Mat &binaryImage, double x_bar, double y_bar, double &theta);	    // 3
int get_moments(Mat &binary_image, vector<vector<Point> > contours, vector<double> &raw_moments, vector<double> &hu_moments);
Mat reorder_eigenvectors(Mat& input_mat);
void reduce_feature_dimension(Mat& reordered_eigen_vectors, Mat& trans_grouped_vector, Mat& reduced_grouped_feature_vector);

// Other functions
int perform_k_means(Mat& feature_points, Mat& feature_mean);
Mat perform_hermitz_function_approximation(const Mat& feature_vector);
#endif // VIDEO_PROCESSING_LIB_HPP

