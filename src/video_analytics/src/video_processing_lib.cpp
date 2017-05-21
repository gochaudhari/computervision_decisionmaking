#include <iostream>
#include <cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <stdio.h>
#include <video_analytics/video_processing_lib.hpp>
#include <math.h>
#include <fstream>

#define PIZZA_PI 3.14159265

using namespace std;
using namespace cv;

extern int number_of_samples, number_of_features, number_of_reduced_features;

/// ROUTINE FOR FEATURE CALCULATION.
/// RETURN: FUNCTION FEATURES VECTOR: Mat&
void feature_calculation(Mat& grayImage, Mat& binaryImage, ofstream& TextFile, bool save_to_file, Mat &feature_vector) {
	// Feature parameter initialization
	int perimeter = 0;
	double area = 0, x_bar = 0, y_bar = 0, theta = 0;
	vector<vector<Point> > contour;
	vector<double> hu_moments(7, 0);
	vector<double> raw_moments(7, 0);
	int feature_counter = 0;

	/// FEATURE 1: PERIMETER
	/// FEATURE 2:	AREA
	// Collect data in perimeter and area
	cout << "Calculating PERIMETER and AREA" << endl;
	find_contour_features(grayImage, perimeter, area, contour);
	cout << "FEATURE 1: Perimeter : " << perimeter << endl;
	feature_vector.at<double>(0, feature_counter) = perimeter; feature_counter++;
//	feature_points.at<double>(image_counter - 1, feature_counter) = perimeter; feature_counter++;
	cout << "FEATURE 2: Area : " << area << endl;
	feature_vector.at<double>(0, feature_counter) = area; feature_counter++;
	//	feature_points.at<double>(image_counter - 1, feature_counter) = area; feature_counter++;
	if(save_to_file) {
		TextFile << perimeter << ",";
		TextFile << area << ",";
	}
	cout << endl;

	/// FEATURE 3: X_BAR
	/// FEATURE 4: Y_BAR.
	// Collect data in x_bar and y_bar
	get_x_y_bar(binaryImage, x_bar, y_bar, area);
	cout << "FEATURE 3: X_bar: " << x_bar << endl;
	feature_vector.at<double>(0, feature_counter) = x_bar; feature_counter++;
//	feature_points.at<double>(image_counter - 1, feature_counter) = x_bar; feature_counter++;
	cout << "FEATURE 4: Y_bar: " << y_bar << endl;
	feature_vector.at<double>(0, feature_counter) = y_bar; feature_counter++;
//	feature_points.at<double>(image_counter - 1, feature_counter) = y_bar; feature_counter++;
	if(save_to_file) {
		TextFile << x_bar << ",";
		TextFile << y_bar << ",";
	}
	cout << endl;

	/// FEATURE 5: THETA
	get_theta(binaryImage, x_bar, y_bar, theta);
	cout << "FEATURE 5: Theta: " << theta << endl;
	feature_vector.at<double>(0, feature_counter) = theta; feature_counter++;
//	feature_points.at<double>(image_counter - 1, feature_counter) = theta; feature_counter++;
	if(save_to_file) {
		TextFile << theta << ",";
	}
	cout << endl;

	/// FEATURE 6: 7 Raw moments
	/// FEATURE 7: 7 Hu moments
	get_moments(binaryImage, contour, raw_moments, hu_moments);
	cout << "FEATURE 6: Raw Moments: ";
	for(int i = 0; i < raw_moments.size(); i++)
	{
		cout << raw_moments[i] << " ";
		feature_vector.at<double>(0, feature_counter) = raw_moments[i]; feature_counter++;
//		feature_points.at<double>(image_counter - 1, feature_counter) = raw_moments[i]; feature_counter++;
		if(save_to_file) {
			TextFile << raw_moments[i] << ",";
		}
	}
	cout << endl << endl;
	cout << "FEATURE 7: Hu Moments: ";
	for(int i = 0; i < hu_moments.size(); i++)
	{
		cout << hu_moments[i] << " ";
		feature_vector.at<double>(0, feature_counter) = hu_moments[i]; feature_counter++;
//		feature_points.at<double>(image_counter - 1, feature_counter) = hu_moments[i]; feature_counter++;
		if(save_to_file) {
			TextFile << hu_moments[i] << ",";
		}
	}

	cout << endl;
}

void perform_image_preprocessing(Mat& input_image, Mat& gray_image, Mat& binary_image) {
	// Showing the input image. *This is in order to keep an order of all the images
	namedWindow("Input Window", WINDOW_AUTOSIZE);
	imshow("Input Window", input_image);

	// Converting that image into a grayscale image.
	cvtColor(input_image, gray_image, COLOR_RGB2GRAY);

	// Showing the grayscale image. *This is in order to keep an order of all the images
	namedWindow("Grayscale Window", WINDOW_AUTOSIZE);
	imshow("Grayscale Window", gray_image);

	// Coverting a grayscale image into a binary image.
	threshold(gray_image, binary_image, 100, 255, CV_THRESH_BINARY);

	// Showing the binary image.
	namedWindow("Binary Image", WINDOW_AUTOSIZE);
	imshow("Binary Image", binary_image);
}

// Get the input array of features here.
// Return double as mean
double calculate_mean(Mat& features_array) {
	cv::Scalar mean_val = mean(features_array);
	return mean_val[0];
}

// Function used for getting the perimeter of the image.
// to use opencv internal functions
// Return the perimeter and area of the image.
int find_contour_features(Mat &input_image, int &perimeter, double &area, vector<vector<Point> > &contours)
{
	Mat contour_display_mat = Mat::zeros(input_image.rows, input_image.cols, CV_8UC1);
	int threshold = 100;

	/// Detect edges using canny
	Canny( input_image, input_image, threshold, threshold*2, 3 );
	imshow("Canny Output", input_image);								// Brother, don't get confused
																		// input_image is Canny output.
																		// Do not use input_image to display original image

	// Finding different contours
	vector<Vec4i> hierarchy;

	findContours(input_image, contours, hierarchy,
				  CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

	for(int idx = 0 ; idx >= 0; idx = hierarchy[idx][0])
	{
		Scalar color(rand()&255, rand()&255, rand()&255);
		drawContours(contour_display_mat, contours, idx, color, CV_FILLED, 8, hierarchy);
	}

	for(int counter = 0; counter < contours.size(); counter++)
	{
		perimeter = perimeter + arcLength(contours[counter], true);
		area  = area + contourArea(contours[counter]);
	}

	namedWindow("Contour Components", 1);
	imshow("Contour Components", contour_display_mat);

	return 0;
}

// This function is for getting the x bar and y bar features of the image
int get_x_y_bar(Mat &binaryImage, double &x_bar, double &y_bar, int area)
{
	int row_counter, col_counter;

	for(row_counter = 0; row_counter < binaryImage.rows; row_counter++) {
		for(col_counter = 0; col_counter < binaryImage.cols; col_counter++) {
			x_bar = x_bar + (row_counter * (int)binaryImage.at<uchar>(row_counter, col_counter));
			y_bar = y_bar + (col_counter * (int)binaryImage.at<uchar>(row_counter, col_counter));
		}
	}
	x_bar = x_bar/area;
	y_bar = y_bar/area;

	return 0;
}

// Get the elongation axis and the theta value of the image
int get_theta(Mat &binaryImage, double x_bar, double y_bar, double &theta)
{
	double a = 0, b = 0, c = 0;
	int rows = binaryImage.rows, row_counter = 0, cols = binaryImage.cols, col_counter = 0;
	double angle = 0;

	// Calculate a, b, c
	for(row_counter = 0; row_counter < rows; row_counter++) {
		for(col_counter = 0; col_counter < cols; col_counter++) {
			a = a + ((row_counter - x_bar) *
					 (int)binaryImage.at<uchar>(row_counter, col_counter));
			b = b + (2 * (row_counter - x_bar) *
					 (col_counter - y_bar) *
					 (int)binaryImage.at<uchar>(row_counter, col_counter));
			c = c + ((col_counter - y_bar) *
					 (int)binaryImage.at<uchar>(row_counter, col_counter));
		}
	}

	// Calculate value of theta using a, b, c
	angle = (atan((b/(a-c))))/2;
	theta = angle * 180/PIZZA_PI;
}

int get_moments(Mat &binary_image, vector<vector<Point> > contours, vector<double> &raw_moments, vector<double> &hu_moments)
{
	int number_of_moments = 7;
	vector<double> hu_moment_return;

	for(int counter = 0; counter < contours.size(); counter++) {
		cv::Moments mom = cv::moments(contours[counter], false);
		raw_moments[0] += mom.nu20;
		raw_moments[1] += mom.nu11;
		raw_moments[2] += mom.nu02;
		raw_moments[3] += mom.nu30;
		raw_moments[4] += mom.nu21;
		raw_moments[5] += mom.nu12;
		raw_moments[6] += mom.nu03;

		cv::HuMoments(mom, hu_moment_return);
		hu_moments[0] += hu_moment_return[0];
		hu_moments[1] += hu_moment_return[1];
		hu_moments[2] += hu_moment_return[2];
		hu_moments[3] += hu_moment_return[3];
		hu_moments[4] += hu_moment_return[4];
		hu_moments[5] += hu_moment_return[5];
		hu_moments[6] += hu_moment_return[6];
	}
}

Mat reorder_eigenvectors(Mat& input_mat) {
	int counter = 0;
	Mat output_mat;
	Mat col_mat(number_of_features, 1, CV_64F), sorted_indexes(number_of_features, 1, CV_16U);
//	sortIdx();
//	SortFlags;
//ReduceTypes
	for(counter = 0; counter < input_mat.rows; counter++) {
		reduce(input_mat, col_mat, 1, REDUCE_SUM);
	}

	sortIdx(col_mat, sorted_indexes, SORT_ASCENDING | SORT_EVERY_COLUMN);

	for(counter = 0; counter < number_of_reduced_features; counter++) {
		output_mat.push_back(input_mat.row(sorted_indexes.at<int>(counter, 0)));
	}
	return output_mat;
}

void reduce_feature_dimension(Mat& reordered_eigen_vectors, Mat& trans_grouped_vector, Mat& reduced_grouped_feature_vector) {
	reduced_grouped_feature_vector = reordered_eigen_vectors * trans_grouped_vector;
	transpose(reduced_grouped_feature_vector, reduced_grouped_feature_vector);
}

/// TODO: Get all the features from the excel sheet
/// And
int perform_k_means(Mat& feature_points, Mat& feature_mean)
{
	// Open the file for reading the data in it
	// File handling parameters
	ifstream MyExcelFile;
	MyExcelFile.open("/home/gauraochaudhari/Documents/CMPE297/Slider_images_all_one/feature_output.ods");
	std::string line;
	string cell_value, delimiter = ",";

	int number_of_clusters = 5, sample_counter = 0, feature_counter = 0;

	// Creating a mat for
	Scalar colorTab[] =
	{
		Scalar(255,100,100),
		Scalar(255,0,255),
	};
	Mat img(500, 500, CV_8UC3);
	int count = 0;

	//Create a matrix for all the features here
	Mat labels;
	std::string::size_type sz;     // alias of size_t

	while (std::getline(MyExcelFile, line))
	{
		size_t pos = 0;
		feature_counter = 0;
		while ((pos = line.find(delimiter)) != std::string::npos) {
			cell_value = line.substr(0, pos);
			line.erase(0, pos + delimiter.length());

			if(count != 0) {
//				std::cout << cell_value << " ";
//				feature_points.at<double>(sample_counter, feature_counter) = std::stof(cell_value, &sz);
//				cout << feature_points.at<double>(sample_counter, feature_counter) << endl;
			}

			feature_counter++;
		}

		cout << "\nMean: ";
		cout << feature_mean.at<double>(0, sample_counter) << endl << endl;

		count++;
		sample_counter++;
	}


//	Mat feature_points(number_of_samples, 2, CV_64F), labels;

//	feature_points.at<Point_>(0) = Point(0, 0);
//	feature_points.at<Point2f>(1) = Point(1, 0);
//	feature_points.at<Point2f>(2) = Point(0, 1);
//	feature_points.at<Point2f>(3) = Point(1, 1);
//	feature_points.at<Point2f>(4) = Point(2, 1);
//	feature_points.at<Point2f>(5) = Point(1, 2);
//	feature_points.at<Point2f>(6) = Point(2, 2);
//	feature_points.at<Point2f>(7) = Point(3, 2);
//	feature_points.at<Point2f>(8) = Point(6, 6);
//	feature_points.at<Point2f>(9) = Point(7, 6);
//	feature_points.at<Point2f>(10) = Point(8, 6);
//	feature_points.at<Point2f>(11) = Point(6, 7);
//	feature_points.at<Point2f>(12) = Point(7, 7);
//	feature_points.at<Point2f>(13) = Point(8, 7);
//	feature_points.at<Point2f>(14) = Point(9, 7);
//	feature_points.at<Point2f>(15) = Point(7, 8);
//	feature_points.at<Point2f>(16) = Point(8, 8);
//	feature_points.at<Point2f>(17) = Point(9, 8);
//	feature_points.at<Point2f>(18) = Point(8, 9);
//	feature_points.at<Point2f>(19) = Point(9, 9);

	/// TODO: While writing this routine, make sure the number of clusters is either equal to
	/// the number of points or they are less than the number of points.
	/// Can use MIN function to limit the number of clusters

	Mat centers;
	kmeans(feature_points, number_of_clusters, labels,
		TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
		   3, KMEANS_PP_CENTERS, centers);

	cout << "\nShowing Segregated Labels" << endl;
	cout << labels << endl;

	img = Scalar::all(0);
	for(sample_counter = 0; sample_counter < number_of_samples; sample_counter++ )
	{
		int cluster_id = labels.at<int>(sample_counter);
		Point ipt = feature_mean.at<Point>(sample_counter);
//		Point ipt = feature_points.at<Point2f>(sample_counter);
		circle(img, ipt, 5, colorTab[cluster_id], FILLED, LINE_AA );
	}
	imshow("clusters", img);
}

double hermitz_1(double x) {
	return 2*x;
}

double hermitz_2(double x) {
	return 4*(pow(x,2))-2;
}

Mat perform_hermitz_function_approximation(const Mat& feature_vector) {
	int total_terms = 0, counter = 0, order = 0, k_total_val = (number_of_reduced_features-1), k_val_counter = 0;
	int const_term = 0;		// Do not confuse with the initial constant. This is the constant in every kth iteration of 2nd order.
							// Specified later

	for(counter = number_of_reduced_features; counter > 0; counter--) {
		total_terms += counter;
	}
	total_terms = 1 + number_of_reduced_features + total_terms;
	cout << "\nTotal Terms: " << total_terms << endl;

	// Create an array for all these terms
	Mat phi_terms(1, total_terms, CV_64F);
	int phi_counter = 0;

	phi_terms.at<double>(0, 0) = 1;					// 1
	phi_counter++;

	// Since first term is always constant, phi_counter is set to 1 here.
	phi_counter = 1;

	// Constant Value
	cout << "Constant Value" << endl;
	cout << "Term 1" << "->Value: " << phi_terms.at<double>(0, 0) << endl << endl;

	// Calculating 1st order terms
	order = 1;
	cout << "Order " << order << endl;
	for(counter = 0; counter < number_of_reduced_features; counter++) {				// 1, 2, 3, 4
		phi_terms.at<double>(0, phi_counter) = hermitz_1(feature_vector.at<double>(0, counter));
		cout << "Term " << phi_counter+1 << "->Value: " << phi_terms.at<double>(0, phi_counter) << endl;
		phi_counter++;
	}
	cout << endl;

	// Calculating second order terms										// 5, 6, 7 -> 8, 9 -> 10
	order = 2; k_val_counter = 1;
	for(k_val_counter = 1; k_val_counter <= k_total_val; k_val_counter++) {
		cout << "Order:" << order << " K:" << k_val_counter << endl;
		const_term = hermitz_1(feature_vector.at<double>(0, k_val_counter-1));
		for(counter = 0; counter < (number_of_reduced_features-k_val_counter); counter++) {
			phi_terms.at<double>(0, phi_counter) = const_term * hermitz_1(feature_vector.at<double>(0, k_val_counter));
			cout << "Term " << phi_counter+1 << "->Value: " << phi_terms.at<double>(0, phi_counter) << endl;
			phi_counter++;
		}
	}
	cout << endl;

	// Order 2-3
	order = 2;
	cout << "Order " << order << "-Quad" << endl;
	for(counter = 0; counter < number_of_reduced_features; counter++) {				// 1, 2, 3, 4
		phi_terms.at<double>(0, phi_counter) = hermitz_2(feature_vector.at<double>(0, counter));
		cout << "Term " << phi_counter+1 << "->Value: " << phi_terms.at<double>(0, phi_counter) << endl;
		phi_counter++;
	}
	cout << endl;

	// Checking if things worked out properly
	if(phi_counter == total_terms) {
		cout << "Total terms calculated properly." << endl;
	}

	cout << "------------------------------------------------------------------------------------------------" << endl << endl;

	return phi_terms;
}
