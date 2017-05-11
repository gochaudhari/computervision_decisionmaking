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
int number_of_samples = 25, number_of_features = 19, number_of_reduced_features = 10;
vector<double> alpha;

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
	Mat input_image, gray_image, binary_image;
	string activity = argv[1];									// What activity to perform
	int images_per_group = atoi(argv[2]);
	bool is_offline_image = false, is_live_video = false, is_video = false;
	int image_count = number_of_samples;
	int group_counter = 0, number_of_groups = image_count/images_per_group;

	// Variables for image handling
	string image_name, image_format(".png");					// Setting up image directory and image names
	int image_counter = 1;
	/// Excel and Text file handling parameters
	ofstream TextFile;

	// Defining string for all traffix signs
	string right("Right Turn"), left("Left Turn"), speed("Speed Limit"), stop("Stop"), intersection("Intersection");

	Mat feature_vector(1, number_of_features, CV_64F), feature_points, reduced_feature_points;

//	Mat feature_vector(1, number_of_features, CV_64F);
//	Mat feature_points(number_of_samples, number_of_features, CV_64F);
	Mat feature_mean(1, number_of_samples, CV_64F);

	if(!strcmp(activity.c_str(), "offline_image")) {
		is_offline_image = true;
	}
	else if(!strcmp(activity.c_str(), "process_live_video")) {
		is_live_video = true;
	}
	else if(!strcmp(activity.c_str(), "video")) {
		is_video = true;
	}
	else {
		cout << "Error in Usage\nUsage: <exe> <activity>[feature, convolution]" << endl;
	}

	/// In any case, we have to get the features from all the training images. After features, hermitz calculations
	/// are also the requirement
	string image_directory("/home/gauraochaudhari/Documents/CMPE297/Slider_images_all_one/");
	TextFile.open("/home/gauraochaudhari/Documents/CMPE297/Slider_images_all_one/feature_txt.txt");

	for(image_counter = 1; image_counter <= image_count; image_counter++) {
		cout << "\n\nIMAGE NUMBER: " << image_counter << endl;
		image_name = to_string(image_counter);
		image_name = image_name.append(image_format);
		image_name = image_directory.append(image_name);

		cout << image_name << endl;
		input_image = imread(image_name);

		// Checking the image data. If there is any data-OK. If not, error.
		if(!input_image.data)
		{
			cout << "Could not open this image. Please try with any other Image";
			return -1;
		}

		perform_image_preprocessing(input_image, gray_image, binary_image);

		/// CALL A ROUTINE THAT DOES FEATURE EXTRACTION AND THEN RETURNS THE FEATURE VECTOR
		feature_calculation(gray_image, binary_image, TextFile, true, feature_vector);
//		Calculating the mean of the features
//		//feature_vector.copyTo(feature_points[image_counter - 1]);
		feature_points.push_back(feature_vector);

		image_directory = "/home/gauraochaudhari/Documents/CMPE297/Slider_images_all_one/";
		image_format = ".png";

		TextFile << endl;
	}
	TextFile.close();


//	Mat temp(1, 2, CV_64F);
//	for(int i = 0; i < 2; i++) {
//		for(int j = 0; j < 2; j++) {
//			temp.at<double>(i, j) = (++counter)*100;
//			cout << temp.at<double_t>(i, j) << " ";
//		}
//		cout << endl;
//	}

//	for(int i = 0; i < 2; i++) {
//		for(int j = 0; j < 2; j++) {
//			cout << temp.at<double_t>(i, j) << " ";
//		}
//		cout << endl;
//	}
//	temp.at<double>(0, 2) = 32;
//	Now perform k means clustering
//	perform_k_means();

	/// PERFORMING FEATURE VECTOR REDUCTION
	Mat buffer, grouped_feature_vector(images_per_group, number_of_features, CV_64F), trans_grouped_vectors;
	Mat reduced_grouped_feature_vector/*(images_per_group, number_of_reduced_features, CV_64F)*/;
	Mat covariance_mat, mean;
	Mat eigen_values, eigen_vectors, reordered_eigen_vectors(number_of_reduced_features, number_of_features, CV_64F);
	// Find covariance matrix
	int counter = 0, counter_k;
	for(group_counter = 0; group_counter < number_of_groups; group_counter++) {
		for(counter = 0; counter < images_per_group; counter++) {
			for(counter_k = 0; counter_k < number_of_features; counter_k++) {
				grouped_feature_vector.at<double>(counter, counter_k) =
						feature_points.row((images_per_group*group_counter)+counter).at<double>(0, counter_k);
			}
		}

		transpose(grouped_feature_vector, trans_grouped_vectors);
		// Creating Covariance matrix from this grouped_feature_vector
		calcCovarMatrix(grouped_feature_vector, covariance_mat, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
		eigen(covariance_mat, eigen_values, eigen_vectors);

		// Finding the lowest 10 eigen vectors from the main eigen vector mat
		reordered_eigen_vectors = reorder_eigenvectors(eigen_vectors);

		// Reducing feature dimension to 10 features
		reduce_feature_dimension(reordered_eigen_vectors, trans_grouped_vectors, reduced_grouped_feature_vector);

		// Fill the main feature vector from the reduced group vector
		for(counter = 0; counter < images_per_group; counter++) {
			reduced_feature_points.push_back(reduced_grouped_feature_vector.row(counter));
		}
	}
	/// reduced_feature_points contains the reduced feature vectors.


	/// Call the Hermitz function in a loop that spans over all the samples present
	///
	/// Return: Return value of phi's
	///
	/// Calculate the value of alpha after getting the values of phi's;
	Mat phi_values, val;
	for(int group_counter = 0; group_counter < number_of_groups; group_counter++) {
		cout << "Group: " << group_counter + 1 << endl;
		phi_values = Mat::zeros(1, 66, CV_64F);
		cout << phi_values << endl;
		for(image_counter = 0; image_counter < images_per_group; image_counter++) {
			// Getting Hermit polynomial for each image
			cout << "IMAGE " << (images_per_group*group_counter) + image_counter + 1 << endl;
			phi_values += perform_hermitz_function_approximation(feature_points.row(images_per_group*group_counter + image_counter));
		}

		reduce(phi_values, val, 1, REDUCE_SUM);
		// Calculate value of alpha here.
		alpha.push_back((val.at<double>(0,0))/images_per_group);
	}

	cout << alpha[0] << endl;
	cout << alpha[1] << endl;
	cout << alpha[2] << endl;
	cout << alpha[3] << endl;
	cout << alpha[4] << endl;

	if(is_offline_image){
	/// Steps to test on an image
	///
	/// 1) Load an image
	Mat test_image = imread("/home/gauraochaudhari/Documents/CMPE297/Slider_images_all_one/test.jpg");

	/// 2) Do pre-processing on image
	perform_image_preprocessing(test_image, gray_image, binary_image);

	/// 3) Calculate features
	feature_calculation(gray_image, binary_image, TextFile, false, feature_vector); //	false parameter to disable file storage

	/// 4) Reduce the feature vector
	// QUESTION HERE: HOW TO REDUCE THE FEATURE VECTOR DIMENSION


	/// 5) Calculate hermitz values - Get phi from here



	/// 6) Calculate PDF for all alpha




	}
	else if(is_live_video) {

	}
	else if(is_video)
	{
		string video_name = argv[3];
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

	waitKey();
	return 0;
}
