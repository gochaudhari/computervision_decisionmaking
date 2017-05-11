#include <iostream>
#include <video_analytics/video_processing_lib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <fstream>

using namespace std;
using namespace cv;

/// TODO: Main objective for this thing is to achieve some
/// 
int g_slider_position = 0;
CvCapture* capture = NULL;
IplImage* frame;
Mat store_image;
int filenumber = 1;
string filename, image_type_1, image_type_2;
int number_of_samples = 25, number_of_features = 19;
Mat feature_points(number_of_samples, number_of_features, CV_32F);
Mat feature_mean(1, number_of_samples, CV_32F);

void onTrackbarSlide(int pos) {
	cvSetCaptureProperty(
				capture,
				CV_CAP_PROP_POS_FRAMES,
				pos
				);

	store_image = cvarrToMat(frame);

	filename = image_type_1.append(to_string(filenumber));
	filename = filename.append(".jpg");
	imwrite(filename, store_image);
	cout << "Position changed. File stored. Current position: " << pos << endl;

	image_type_1 = image_type_2;
	filename.erase();
	filenumber++;
}

/// TODO: The final requirement of this project is to get the following things done
/// 1) Get the live feed from the video
/// 2) Perform 2D convolution on the image using give kernels
/// 3) Edge detector- Canny edge detector
/// 4) Gray to Binary image
/// 5) Calculate Features and form a feature vector
/// 6) Reduce feature vector dimension
/// 7) Calculate PDF using Hermitz calculation
int main(int argc, char *argv[])
{
	Mat input_image, grayImage, binaryImage;
	string activity = argv[1];
	bool is_manual_feature = false, is_convolution = false, is_feature_calculation = false, is_video = false, is_live_video = false;
	int image_count = number_of_samples;

	string kernel_file;
	string kernel_pattern;
	int feature_vector_dimension, number_of_samples, number_of_groups;

	if(!strcmp(activity.c_str(), "manual_feature"))
	{
		is_manual_feature = true;
		feature_vector_dimension = 3;
		number_of_samples = 4;
		number_of_groups = 2;
	}
	else if(!strcmp(activity.c_str(), "convolution"))
	{
		is_convolution = true;
		// Checking for the input argument image path
		if(argc != 4)
		{
			cout << "Error in Usage\nUsage: <exe> <activity>[feature, convolution] <image_name> <kernel_file_name>" << endl;
			return -1;
		}
		kernel_file = argv[2];
		kernel_pattern = argv[3];
	}
	else if(!strcmp(activity.c_str(), "feature_calculation"))
	{
		is_feature_calculation = true;
	}
	else if(!strcmp(activity.c_str(), "video"))
	{
		is_video = true;
	}
	else if(!strcmp(activity.c_str(), "process_live_video")) {
		is_live_video = true;
	}
	else
	{
		cout << "Error in Usage\nUsage: <exe> <activity>[feature, convolution]" << endl;
	}

	if(is_feature_calculation)
	{
		int sample_counter = 0, feature_counter = 0;

		/// Setting up image directory and image names
		string image_name, image_format(".jpg");
		string image_type = argv[2];
		int image_counter = 1;
		string image_directory("/home/gauraochaudhari/Documents/CMPE297/Slider_images_all_one/");

		/// Excel and Text file handling parameters
		ofstream MyExcelFile, TextFile;
		MyExcelFile.open("/home/gauraochaudhari/Documents/CMPE297/Slider_images_all_one/feature_output.ods");
		TextFile.open("/home/gauraochaudhari/Documents/CMPE297/Slider_images_all_one/feature_txt.txt");

		// Handling the file implementation here
		MyExcelFile << endl;
		MyExcelFile << endl;
		MyExcelFile << endl;

		MyExcelFile << "," << "Perimeter";
		MyExcelFile << "," << "Area";
		MyExcelFile << "," << "X_Bar";
		MyExcelFile << "," << "Y_Bar";
		MyExcelFile << "," << "Theta";
		MyExcelFile << "," << "Raw Moment_1";
		MyExcelFile << "," << "Raw Moment_2";
		MyExcelFile << "," << "Raw Moment_3";
		MyExcelFile << "," << "Raw Moment_4";
		MyExcelFile << "," << "Raw Moment_5";
		MyExcelFile << "," << "Raw Moment_6";
		MyExcelFile << "," << "Raw Moment_7";

		MyExcelFile << "," << "Hu Moment_1";
		MyExcelFile << "," << "Hu Moment_2";
		MyExcelFile << "," << "Hu Moment_3";
		MyExcelFile << "," << "Hu Moment_4";
		MyExcelFile << "," << "Hu Moment_5";
		MyExcelFile << "," << "Hu Moment_6";
		MyExcelFile << "," << "Hu Moment_7";

		MyExcelFile << "," << endl;

		while(image_counter <= image_count){
			feature_counter = 0;
			cout << "IMAGE NUMBER: " << image_counter << endl;
			image_name = image_type.append(to_string(image_counter));
			image_name = image_name.append(image_format);
			image_name = image_directory.append(image_name);

			cout << image_name << endl;
			input_image = imread(image_name);

			// Parameter initialization
			int perimeter = 0;
			double area = 0, x_bar = 0, y_bar = 0, theta = 0;
			vector<vector<Point> > contour;
			vector<double> hu_moments;
			vector<double> raw_moments;

			// Checking the image data. If there is any data-OK. If not, error.
			if(!input_image.data)
			{
				cout << "Could not open this image. Please try with any other Image";
				return -1;
			}

			// Showing the input image. *This is in order to keep an order of all the images
			namedWindow("Input Window", WINDOW_AUTOSIZE);
			imshow("Input Window", input_image);

			// Converting that image into a grayscale image.
			cvtColor(input_image, grayImage, COLOR_RGB2GRAY);

			// Showing the grayscale image. *This is in order to keep an order of all the images
			namedWindow("Grayscale Window", WINDOW_AUTOSIZE);
			imshow("Grayscale Window", grayImage);

			// Coverting a grayscale image into a binary image.
			threshold(grayImage, binaryImage, 100, 255, CV_THRESH_BINARY);

			// Showing the binary image.
			namedWindow("Binary Image", WINDOW_AUTOSIZE);
			imshow("Binary Image", binaryImage);

			/// FEATURE 1: PERIMETER
			/// FEATURE 2:	AREA
			// Collect data in perimeter and area
			cout << "Calculating PERIMETER and AREA" << endl;
			find_contour_features(grayImage, perimeter, area, contour);
			cout << "FEATURE 1: Perimeter : " << perimeter << endl;
			feature_points.at<double>(image_counter - 1, feature_counter) = perimeter; feature_counter++;
			cout << "FEATURE 2: Area : " << area << endl;
			feature_points.at<double>(image_counter - 1, feature_counter) = area; feature_counter++;
			MyExcelFile << "," << perimeter;
			MyExcelFile << "," << area;
			TextFile << "," << perimeter;
			TextFile << "," << area;
			cout << endl;

			/// FEATURE 3: X_BAR
			/// FEATURE 4: Y_BAR.
			// Collect data in x_bar and y_bar
			get_x_y_bar(binaryImage, x_bar, y_bar, area);
			cout << "FEATURE 3: X_bar: " << x_bar << endl;
			feature_points.at<double>(image_counter - 1, feature_counter) = x_bar; feature_counter++;
			cout << "FEATURE 4: Y_bar: " << y_bar << endl;
			feature_points.at<double>(image_counter - 1, feature_counter) = y_bar; feature_counter++;
			MyExcelFile << "," << x_bar;
			MyExcelFile << "," << y_bar;
			TextFile << "," << x_bar;
			TextFile << "," << y_bar;
			cout << endl;

			/// FEATURE 5: THETA
			get_theta(binaryImage, x_bar, y_bar, theta);
			cout << "FEATURE 5: Theta: " << theta << endl;
			feature_points.at<double>(image_counter - 1, feature_counter) = theta; feature_counter++;
			MyExcelFile << "," << theta;
			TextFile << "," << theta;
			cout << endl;

			/// FEATURE 6: 7 Raw moments
			/// FEATURE 7: 7 Hu moments
			get_moments(binaryImage, raw_moments, hu_moments);
			cout << "FEATURE 6: Raw Moments: ";
			for(int i = 0; i < raw_moments.size(); i++)
			{
				cout << raw_moments[i] << " ";
				feature_points.at<double>(image_counter - 1, feature_counter) = raw_moments[i]; feature_counter++;
				MyExcelFile << "," << raw_moments[i];
				TextFile << "," << raw_moments[i];
			}
			cout << endl << endl;
			cout << "FEATURE 7: Hu Moments: ";
			for(int i = 0; i < hu_moments.size(); i++)
			{
				cout << hu_moments[i] << " ";
				feature_points.at<double>(image_counter - 1, feature_counter) = hu_moments[i]; feature_counter++;
				MyExcelFile << "," << hu_moments[i];
				TextFile << "," << hu_moments[i];
			}
			cout << endl;

			image_directory = "/home/gauraochaudhari/Documents/CMPE297/Slider_images_all_one/";
			image_type = argv[2];
			image_format = ".jpg";

			MyExcelFile << endl;
			TextFile << endl;

			// Calculating the mean of the features
			feature_mean.at<double>(1, image_counter-1) = calculate_mean(feature_points.row(image_counter-1))/1000;
			image_counter++;
		}

		// We are getting the features here.
		// Now perform k means clustering
//		perform_k_means();
		perform_hermitz_function_approximation();

		MyExcelFile.close();
		TextFile.close();
	}
	else if(is_manual_feature)
	{
		double **mean_result;

		// Complex description of the feature array
		int ***X;

		// Allocating memory for the groups (i will be always number of groups
		/// TODO: Handle memory leak later
		X = (int ***)malloc(number_of_groups * sizeof(int **));

		mean_result = (double **)malloc(number_of_groups * sizeof(double *));

		for(int counter = 0; counter < number_of_groups; counter++)
		{
			mean_result[counter] = (double *)malloc(feature_vector_dimension * sizeof(double));
		}

		// Allocating memory for the number of samples (j will always be number of samples)
		/// TODO: Handle memory leak later
		for(int j = 0; j < number_of_groups; j++)
		{
			X[j] = (int **)malloc(number_of_samples * sizeof(int *));

			//Allocating memory for each array[i] with the number feature vector dimension
			// (k should always be feature vector dimension
			for(int k = 0; k < number_of_samples; k++)
			{
				X[j][k] = (int *)malloc(feature_vector_dimension * sizeof(int));
			}
		}

		/// TODO: Remove this after the feature vectors are filled
		// Temporary: Filling up group 1
		X[0][0][0] = 0; X[0][0][1] = 0; X[0][0][2] = 0;
		X[0][1][0] = 1; X[0][1][1] = 0; X[0][1][2] = 0;
		X[0][2][0] = 1; X[0][2][1] = 0; X[0][2][2] = 1;
		X[0][3][0] = 1; X[0][3][1] = 1; X[0][3][2] = 0;

		// Temporary: Filling up group 2
		X[1][0][0] = 0; X[1][0][1] = 0; X[1][0][2] = 1;
		X[1][1][0] = 0; X[1][1][1] = 1; X[1][1][2] = 0;
		X[1][2][0] = 0; X[1][2][1] = 1; X[1][2][2] = 1;
		X[1][3][0] = 1; X[1][3][1] = 1; X[1][3][2] = 1;

		// Calculation of mean ahead of this
		calculate_mean(X, mean_result, number_of_groups, number_of_samples, feature_vector_dimension);

		// Calculation of covariance
		calculate_covariance(X, number_of_groups, number_of_samples, feature_vector_dimension);

		// Freeing everything
		free(X);
		X = NULL;
		free(mean_result);
		mean_result = NULL;
	}
	else if(is_video)
	{
		string video_name = argv[2];
		image_type_1 = argv[3];
		image_type_2 = argv[3];
		cout << "Starting video" << endl;
		namedWindow("Example 3", CV_WINDOW_AUTOSIZE);
		capture = cvCreateFileCapture(video_name.c_str());

		int frames = (int) cvGetCaptureProperty(
					capture,
					CV_CAP_PROP_FRAME_COUNT
					);
		if( frames!= 0 ) {
			cvCreateTrackbar(
						"Position",
						"Example 3",
						&g_slider_position,
						frames,
						onTrackbarSlide
						);
		}
		while(1)
		{
			frame = cvQueryFrame(capture);
			if( !frame ) break;
			cvShowImage("Example 3", frame);
			char c = cvWaitKey(33);
			if( c == 27 ) break;
		}
		cvReleaseCapture(&capture);
		cvDestroyWindow("Example 3");
	}

	waitKey(0);
	return 0;
}
