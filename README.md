## ObjectDetect

This project is based off of opencv objectdetection.cpp and this git https://github.com/niconielsen32/ComputerVision/blob/master/OpenCVdnn/objectDetection.cpp

This program takes in images and attempts to identify objects in the images, the objects it can detect are displayed in object_detection_classes_coco.txt.
it then draws a rectange over the detected objected and displays its percieved fps and confidence on that object.

The program runs twice, first on your gpu and then on your cpu to compare execution times

Artifacts can be seen if you uncomment lines 134 & 135

## Project structure

This project contains all necessary visual studio files in the uppermost level, the folder named iomage box contains commonplace images to bed fed to the dnn


## Running the project

To run the project click run at the top of visual studio, if you would like to see the images being show, uncomment lines 133 and 134!
Execution time will be displayed in the opened terminal and image output will be in a seperate window, an example is provided below
![cat](https://user-images.githubusercontent.com/14183759/231908266-4de5d8ce-0a03-4803-a221-7133bc689d11.png)

![DetectedCat](https://user-images.githubusercontent.com/14183759/231908316-70e15a4d-4e02-4fbf-9b24-5b4de36b895c.png)

![terminal](https://user-images.githubusercontent.com/14183759/231908678-d83e3304-b0cc-47bf-be7f-33f0a30d800e.png)


## Dependencies and Prereqs

This project depends on openCV with cuda and cuDNN, cudaToolKit these can be installed utilizing cmake GUI and Visual Studio.

opencv source: https://github.com/opencv/opencv
cmake: https://cmake.org/
VisualStudio: https://visualstudio.microsoft.com/
opencv contrib source: https://github.com/opencv/opencv_contrib

Installing opencv with cuda and cuDNN can be quite a challenge, if you would like to install them I recommend watching this video

https://www.youtube.com/watch?v=-GY2gT2umpk

however the general installation is done via the following.

1. downloading opencv source, opencv contrib, cuDNN, cudaToolKit, visual studio and cmake GUI
2. Installing cmake GUI, cudaToolKit, and visual studio with their respective installers 
3. Configuring the installation to include things like cuDNN, fast math, opencvWorld, etc
4. Building and Installing this in cmake GUI
5. Opening the generated opencv.sln file in visual studio
6. Installing the rest of the modules via INSTALL.vcxproj

## Additional notes

My GPU is a 3070ti and my cpu is a i7 9700k for reference
with these components the execution time for my GPU is 15.7 seconds and the execution time on my CPU is 25 seconds!

