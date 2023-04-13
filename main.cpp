/*
Based on opencv's objectdetection.cpp 
Also Based on https://github.com/niconielsen32/ComputerVision/blob/master/OpenCVdnn/objectDetection.cpp

This program takes in images and attempts to identify objects in the images, the objects it can detect are displayed in object_detection_classes_coco.txt.
it then draws a rectange over the detected objected and displays its percieved fps and confidence on that object.

The program runs twice, first on your gpu and then on your cpu to compare execution times

Artifacts can be seen if you uncomment lines 134 & 135
*/

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <string>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace std::chrono;
namespace fs = std::filesystem;


int main(int, char**) {

    for (int i = 0; i < 2; i++) {
        try {
            // init variables
            string imageFilePath = "imageBox/";
            string filename;
            vector<string> class_names;
            ifstream ifs(string("object_detection_classes_coco.txt").c_str());
            string line;
            int counter = 1;

            // Load in all the classes from the file
            while (getline(ifs, line))
            {
                class_names.push_back(line);
            }


            // Read in the neural network from the files
            // Read in trained model and object list
            auto net = readNet("frozen_inference_graph.pb",
                "ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt", "TensorFlow");

            // Runs on GPU during the First iteration, else run on the cpu
            if (i == 0) {
                // Run on GPU
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            }
            else {
                 // run on CPU
                 net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                 net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }

             // Set a min confidence score for the detections
            // confidence is a quantified value determining how much the DNN thinks the object
            // is what it says it is.
            float min_confidence_score = 0.5;


            // starting clock to calculate execution time
            auto start = high_resolution_clock::now();

            // Loop over all entities in the imageBox directory
            for (const auto& entry : fs::directory_iterator(imageFilePath)) {

                cout << entry.path().string() << std::endl;

                // starting execution timer
                auto start = getTickCount();

                // Load in an image
                Mat image = imread(entry.path().string());

                // set the amount of columns and rows in the image
                int image_height = image.cols;
                int image_width = image.rows;



                // Create a blob from the image
                // blobs are used as the input data passed to the dnn
                Mat blob = blobFromImage(image, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5),
                    true, false);


                // Set the blob to be input to the neural network
                net.setInput(blob);

                // Forward pass of the blob through the neural network to get the predictions
                Mat output = net.forward();




                // Matrix with all the detections
                Mat results(output.size[2], output.size[3], CV_32F, output.ptr<float>());

                // Run through all the predictions
                for (int i = 0; i < results.rows; i++) {
                    int class_id = int(results.at<float>(i, 1));
                    float confidence = results.at<float>(i, 2);

                    // Check if the detection is over the min threshold and then draw bbox
                    if (confidence > min_confidence_score) {
                        int bboxX = int(results.at<float>(i, 3) * image.cols);
                        int bboxY = int(results.at<float>(i, 4) * image.rows);
                        int bboxWidth = int(results.at<float>(i, 5) * image.cols - bboxX);
                        int bboxHeight = int(results.at<float>(i, 6) * image.rows - bboxY);
                        rectangle(image, Point(bboxX, bboxY), Point(bboxX + bboxWidth, bboxY + bboxHeight), Scalar(0, 0, 255), 2);
                        string class_name = class_names[class_id - 1];
                        putText(image, class_name + " " + to_string(int(confidence * 100)) + "%", Point(bboxX, bboxY - 10), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 2);
                    }
                }

                // pausing timer as the image is being passed back from the dnn as output
               // and the image detection rectange is drawn.
                auto end = getTickCount();
                auto totalTime = (end - start) / getTickFrequency();


                //imshow("image", image);
                //int k = waitKey(200);

            }

            // end clock and calculate execution time
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);

            if (i == 0) {
                cout << "execution time on GPU is " << (float)(duration.count()) / 1000000 << " seconds\n";
            }
            else {
                cout << "execution time on CPU is " << (float)(duration.count()) / 1000000 << " seconds\n";
            }

        }
        catch (cv::Exception& e)
        {
            cerr << e.msg << endl; // output exception message
        }
    }
}