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
	bool is_offline_image = false, is_live_video = false;
	int image_count = number_of_samples;
	int group_counter = 0, number_of_groups = image_count/images_per_group;

	// Variables for image handling
	string image_name, image_format(".png");					// Setting up image directory and image names
	int image_counter = 1;
	/// Excel and Text file handling parameters
	ofstream TextFile;

	// Defining string for all traffix signs
	vector<double> pdf_values(number_of_groups);
	Mat decision(1, number_of_groups, CV_16U);					// These decision values are in the following order
	string left("Left Turn"), right("Right Turn"), speed("Speed Limit"), stop("Stop"), intersection("Intersection"), decision_text;


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

	/// PERFORMING FEATURE VECTOR REDUCTION
	Mat buffer, grouped_feature_vector(images_per_group, number_of_features, CV_64F), trans_grouped_vectors;
	Mat reduced_grouped_feature_vector/*(images_per_group, number_of_reduced_features, CV_64F)*/;
	Mat covariance_mat, mean;
	Mat eigen_values, eigen_vectors, reordered_eigen_vectors(number_of_reduced_features, number_of_features, CV_64F);

	// Form the multi-channel eigen vector containing matrix
//	Mat all_reordered_eigen_vectors(number_of_reduced_features, number_of_features, CV_64FC(number_of_samples));
	vector<Mat> all_reordered_eigen_vectors;

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
		// At this point, store the eigen vectors in the multi channel mat that stores all the eigen vectors
		all_reordered_eigen_vectors.push_back(reordered_eigen_vectors);

		// Reducing feature dimension to 10 features
		reduce_feature_dimension(reordered_eigen_vectors, trans_grouped_vectors, reduced_grouped_feature_vector);

		// Fill the main feature vector from the reduced group vector
		for(counter = 0; counter < images_per_group; counter++) {
			reduced_feature_points.push_back(reduced_grouped_feature_vector.row(counter));
		}
	}
	/// reduced_feature_points: Reduced Feature Vectors.

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
			phi_values += perform_hermitz_function_approximation(reduced_feature_points.row(images_per_group*group_counter + image_counter));
		}

		reduce(phi_values, val, 1, REDUCE_SUM);
		// Calculate value of alpha here.
		alpha.push_back((val.at<double>(0,0))/images_per_group);
	}

	// Most important part of the code
	cout << alpha[0] << endl;
	cout << alpha[1] << endl;
	cout << alpha[2] << endl;
	cout << alpha[3] << endl;
	cout << alpha[4] << endl;

	if(is_offline_image){
		image_name = argv[3];
		image_name = image_directory.append(image_name);

		/// Steps to test on an image
		///
		/// 1) Load an image
//		Mat test_image = imread("/home/gauraochaudhari/Documents/CMPE297/Original_images/test.jpg");
		Mat test_image = imread(image_name);
		Mat red_feature_vec, trans_feature_vector, re_eigen_vector, hermitz_op;

		/// 2) Do pre-processing on image
		perform_image_preprocessing(test_image, gray_image, binary_image);

		/// 3) Calculate features
		feature_calculation(gray_image, binary_image, TextFile, false, feature_vector); //	false parameter to disable file storage
		transpose(feature_vector, trans_feature_vector);

		/// 4) Reduce the feature vector
		// QUESTION HERE: HOW TO REDUCE THE FEATURE VECTOR DIMENSION
		// This feature vector is reduced using the eigen vectors from all the groups
		for(int counter = 0; counter < number_of_groups; counter++) {
			re_eigen_vector = all_reordered_eigen_vectors[counter];
			reduce_feature_dimension(re_eigen_vector, trans_feature_vector, red_feature_vec);	// red_feature_vec will get the new feature
																								// vector

			// Now finding the Hermitz using this new feature vector
			// Calculating PDF for these 5 feature vectors for this image
			// Using these <number_of_samples> different PDF's
			hermitz_op = perform_hermitz_function_approximation(red_feature_vec);
			reduce(hermitz_op, val, 1, REDUCE_SUM);

			// Calculating PDF now for the appropriate alpha and then storing those five values in a Mat
			// Pushing that value to decision_values mat
			pdf_values[counter] = val.at<double>(0,0) * alpha[counter];
		}
		// Sorting pdf_values
		sortIdx(pdf_values, decision, SORT_DESCENDING | SORT_EVERY_ROW);

		cout << "\n\n----------------------------------------------------------------------------------" << endl;
		cout << "\t\t\t  TESTING ON A NEW IMAGE" << endl;
		cout << pdf_values[0] << " " << pdf_values[1] << " " << pdf_values[2] << " " << pdf_values[3] << " " << pdf_values[4] << endl;
		cout << decision << endl;
		cout << "Detected Image: ";
		switch (decision.at<int>(0,0)) {
			case 0:
				cout << left << endl; break;
			case 1:
				cout << right << endl; break;
			case 2:
				cout << speed << endl; break;
			case 3:
				cout << stop << endl; break;
			case 4:
				cout << intersection << endl; break;
		}
		cout << "----------------------------------------------------------------------------------" << endl;
	}
	else if(is_live_video) {
		// Get the video capture from the the video file.
		cout << "Running from a live video" << endl;
		Mat test_image;
		VideoCapture video_capture_file;
		video_capture_file.open("/home/gauraochaudhari/Documents/CMPE297/Photos and Videos/left.MOV");

		if(!video_capture_file.isOpened()) {
			cout << "Problem opening the video" << endl;
			return -1;
		}
		else {
			cout << "Video Opened" << endl;
		}

		while(video_capture_file.grab()) {
			if(!video_capture_file.retrieve(test_image)) cout << "Image not retrieved" << endl;

			Mat red_feature_vec, trans_feature_vector, re_eigen_vector, hermitz_op;

			/// 2) Do pre-processing on image
			perform_image_preprocessing(test_image, gray_image, binary_image);

			/// 3) Calculate features
			feature_calculation(gray_image, binary_image, TextFile, false, feature_vector); //	false parameter to disable file storage
			transpose(feature_vector, trans_feature_vector);

			/// 4) Reduce the feature vector
			// QUESTION HERE: HOW TO REDUCE THE FEATURE VECTOR DIMENSION
			// This feature vector is reduced using the eigen vectors from all the groups
			for(int counter = 0; counter < number_of_groups; counter++) {
				re_eigen_vector = all_reordered_eigen_vectors[counter];
				reduce_feature_dimension(re_eigen_vector, trans_feature_vector, red_feature_vec);	// red_feature_vec will get the new feature
																									// vector

				// Now finding the Hermitz using this new feature vector
				// Calculating PDF for these 5 feature vectors for this image
				// Using these <number_of_samples> different PDF's
				hermitz_op = perform_hermitz_function_approximation(red_feature_vec);
				reduce(hermitz_op, val, 1, REDUCE_SUM);

				// Calculating PDF now for the appropriate alpha and then storing those five values in a Mat
				// Pushing that value to decision_values mat
				pdf_values[counter] = val.at<double>(0,0) * alpha[counter];
			}
			// Sorting pdf_values
			sortIdx(pdf_values, decision, SORT_DESCENDING | SORT_EVERY_ROW);

			cout << "\n\n----------------------------------------------------------------------------------" << endl;
			cout << "\t\t\t  TESTING ON A NEW IMAGE" << endl;
			cout << pdf_values[0] << " " << pdf_values[1] << " " << pdf_values[2] << " " << pdf_values[3] << " " << pdf_values[4] << endl;
			cout << decision << endl;
			cout << "Detected Image: ";
			switch (decision.at<int>(0,0)) {
				case 0:
					decision_text = left; break;
				case 1:
					decision_text = right; break;
				case 2:
					decision_text = speed; break;
				case 3:
					decision_text = stop; break;
				case 4:
					decision_text = intersection; break;
			}
			cout << decision_text << endl;
			putText(test_image, decision_text, cvPoint(30,30),
				FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,255), 1, CV_AA);
			imshow("Input Image", test_image);
			cout << "----------------------------------------------------------------------------------" << endl;

			char c = cvWaitKey(33);
		}

		video_capture_file.release();
	}

	waitKey();
	return 0;
}
