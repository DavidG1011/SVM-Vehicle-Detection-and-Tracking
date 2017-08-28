**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hogexample.png
[image2]: ./output_images/1scaletest6.png
[image3]: ./output_images/heatmapexample.png
[image4]: ./output_images/multidrawnboxes.png
[image5]: ./output_images/normalized.png
[image6]: ./output_images/spacialexample.png
[image7]: ./output_images/spacialhog.png
[image8]: ./output_images/test.png
[image9]: ./output_images/test2.png
[image10]: ./output_images/test3.png
[image11]: ./output_images/carnoncar.png
[image12]: ./output_images/ycr.png
[video1]: ./project_video.mp4

---
**[Rubric](https://review.udacity.com/#!/rubrics/513/view) Points:**
---


**Histogram of Oriented Gradients (HOG):**

The code to extact HOG features is contained in code cell 4 of the IPython notebook. Namely in the ```get_hog_features``` function. 
This function passes in ```orient, pix_per_cell``` and ```cell_per_block``` as paramaters and also has a boolean paramater to chose to return the HOG applied image. This function is simply a "handler" for the sklearn ```hog()``` function. 

With the HOG paramaters I was simply striving for performance, as in: I wanted to achieve the best LinearSVC accuracy I could. After experimenting with these paramaters I found that these were the best set of paramaters for my purpose:

| Orient | Pixels per cell | Cells per block |
|:------:|:---------------:|:---------------:|
|  9      | 8              | 2               |


Here is an example of a HOG image from a hood camera perspective:

![alt text][image1]


And here is an example of a single car 


---


**Spatial Binning:**


Another feature I used was to resize images to still retain useful color data, but increase the generalization of the classifier. 
I settled on spatially binning my images to a size of 32,32. This code is located in code cell 4, function : ```spatial_bin()```

Example Image:

![alt text][image6]

As can be seen above, the image still retains its qualities fairly well and it is easy to determine that it is a picture of a car. 


---


**Color Spaces and Histograms:**

Before spacial binning, the image is converted into a different color space. In experimenting, I found that the 'YCrCb' color space provided better outlier elimination while still retaining correct predictions than other color spaces I experimented with. Other spaces explored were LUV, HSV, YUV, and HLS.

Here is an example of a YCrCb color space converted car:


![alt text][image12]


After getting a color converted spacial bin, I use the function ```color_hist()``` to create histograms for each color channel of the image passed into it. These channels are then concatenated into one feature vector. 


---


**Training the Classifier and Predicting**

I started by compiling a list of all the `vehicle` and `non-vehicle` image names to be read in later. This list was created in code cell 3. The final specs of this set were as follows:


* Car list: 8792 images.
* Non-car list: 8968 images.
* Image shape: (64, 64, 3)


Here is a composite example of one of each of the `vehicle` and `non-vehicle` classes:


![alt text][image11]



After I enumerated a list of images, I used my function ```extract_features()```--Also located in code cell 4, to apply the above steps to each image in the set. This function reads in all images from the car and non-car dataset, and then calls the previously discussed functions: ```spatial_bin()``` , ```get_hog_features``` and ```color_hist``` , then concatenates all those individual features into one feature vector that is effective for training the classifer. The function call for extracting is in code cell 5 of the IPython notebook.

The classifier I used is a "Linear Support Vector Classifier" or a LinearSVM. I also experimented with a Support Vector Machine, and while the accuracy was favorable, it was just too slow for training and predicting to be entirely effective. Before training, my features were scaled and normalized to reduce the broadness and variance of the data. This can be seen in code cell 6 of the IPython notebook.

Here is a histogram value comparison of a raw image from the car dataset, and that same image normalized and scaled:


![alt text][image5]


The code for this histogram is located in code cell 7 of the IPython notebook.

Before training, I used Numpy functions to create labels for each of the car and non-car sets. A label of 1 being a car image and a label of 0 being a non-car image. The code for this is in code cell 8 of the IPython notebook. The features and labels were then split into a training and testing set to be used with the classifer. The final specs of the classifier are shown below:

* Feature vector length: 8460
* SVC Trained in: Average 20~ seconds (This would vary)
* Accuracy: 99.61% (Would also vary)

The code for classifying and predicting is included in code cell 10 of the IPython notebook.

---



###Sliding Window Search



I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

