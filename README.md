**Project Code: [Here](https://github.com/DavidG1011/Udacity---Vehicle-Detection-and-Tracking/blob/master/Project.ipynb)**


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
[image13]: ./output_images/singlemap.png
[image14]: ./output_images/test6.png
[video1]: ./project_video.mp4

---
**[Rubric](https://review.udacity.com/#!/rubrics/513/view) Points:**
---


**Histogram of Oriented Gradients (HOG):**

The code to extact HOG features is contained in code cell 4 of the IPython notebook. Namely in the ```get_hog_features``` function. 
This function passes in ```orient, pix_per_cell``` and ```cell_per_block``` as paramaters and also has a boolean paramater to chose to return the HOG applied image. This function is simply a "handler" for the sklearn ```hog()``` function. 

With the HOG paramaters I was simply striving for performance, as in, I wanted to achieve the best LinearSVC accuracy I could. After experimenting with these paramaters I found that these were the best set of paramaters for my purpose:

| Orient | Pixels per cell | Cells per block |
|:------:|:---------------:|:---------------:|
|  9      | 8              | 2               |


Here is an example of a HOG image from a hood camera perspective:


![alt text][image1]


And here is an example of a single car:


![alt text][image7]


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


After getting a color converted spacial bin, I use the function ```color_hist()``` to create histograms for each color channel of the image passed into it. These channels are then concatenated into one feature vector. This function can be found in code cell 4 of the IPython notebook.


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
* Accuracy: 99.61% (Would also vary - unfortunately)

The code for classifying and predicting is included in code cell 10 of the IPython notebook.

---


**Sliding Window Search:**

This concept was implemented in function ```find_cars()``` located in code cell 4 of the IPython notebook. 

The first thing I wanted to do with this approach was eliminate as many outliers as possible, as I knew I wanted to do a multi-scale sliding window to detect cars at different distances from the camera. Any outliers during this approach would just iterate over and over, making themselves a nuisance during thresholding as they would then be reasonable "car" detections. 

I decided on a good baseline minimum y search position and minimum postion to focus only on the road by printing out one of the test images into a ```matplotlib``` interactive window and guesstimating a good cutoff point. The final points I decided on were ```ystart = 400``` and ```ystop = 650```. ```ystart``` being the minimum y value to start searching in the image and the latter the maximum y value to stop searching. The function used ```find_cars``` (Adapted from the Udacity lessons) does most of the heavy lifting in the car detection process. This function follows the path of:

* Takes in desired image (Is fed stream of images from video during pipeline phase)
* Crops image based on y starting point and stopping point
* Converts color space of image (Same as what images were traing with)
* Rescale image if desired scale is not 1 (Allows for detecting cars closer or further away from camera)
* Defines pixel blocks and steps to take based on ```pix_per_cell``` and ```cell_per_block``` paramaters.
* Computes hog features for each color channel
* (For each cell in block) Gets hog for current window
* Extracts the current window
* Gets spacial binning and color histogram features for each frame
* Predicts if car or non-car using trained classifer
* Ignores non-car predictions or draws bounding boxes on predicted car positions in original image

(This code can be fully read with comments in cell 4 of the IPython notebook, function ```find_cars()```)

In simpler terms, this function essentially creates a grid of the defined image area and captures each of those grid pieces to be feature extracted and classified. This follows the same path as the classifier feature extraction to ensure that the new images to be predicted from the video stream or otherwise can be correctly identified. 

---

**Heat Mapping:**

The sliding window approach is great for detection, but has the consequence of sometimes having multiple grids being detected as an individual car, or rather, it just doesn't track the previous found bounding box to consider it might be the same car.


This can be seen here where multiple detection bounding boxes are plotted for the same car: 



![alt text][image4]



To remedy this, "heat" mapping can be used to detect where many bounding boxes appear in an image. By adding "heat" or a 1 where pixels are within a bounding box, a heat map can be created to represent the highest concentration of bounding boxes, which will likely be where a car is. This is also helpful for getting rid out outliers, as you can threshold how many pixels need to be present for it to be an accurate prediction. The functions for this are in code cell 4 of the IPython notebook. Namely functions ```add_heat()``` and ```apply_threshold```. This implementation is explored in code cells [13 - 15] of the IPython notebook. 

Here is the heat map for the previously shown bounding boxes:


![alt text][image13]


By taking the outermost corners of the bounding heat mapped boxes, you can plot the multiple bounding boxes as 1 bounding box. The function for this is ```draw_labeled_bboxes()``` located in code cell 4 of the IPython notebook. 

This results in:

![alt text][image2]


This heat map approach can also be applied when using a muli-scale approach, as in the grid size for the ```find_cars``` function is changed with the ```scale``` paramater to achieve different vehicle detections. This is good for making sure all vehicles in your frame are detected at different distances. These different scaled detections are appended to a list and then heat mapped to achieve a similar result to the previous example. To determine which scales worked best and what range I wanted, I simply tried different scales and wrote down which ones looked best. Not entirely scientific, but it gets the job done. I decided upon a range of [1.4 - 2.0] with a 0.2 step increment. This seemed to still get good results while eliminating outliers. Anything below 1 for scale gave me too many outliers, and anything above about 2.2 or so produced little to no results. 

Applying this appraoch to the same image, while computationally costly, imporved car detection greatly for me. The function used for this is ```multi_scale``` located in code cell 4. 


Here is a heat map of the same image but with a multi-scale bounding box detection:


![alt text][image3]


And the ideal single bounding box:


![alt text][image14]


Much better.

---


**Optimization:**

To optimize classification: I played with many different color values, spatial bin sizes, heat thresholds, and scales, as mentioned above. I also explored the ever frustrating world of: "Just because my classifier is accurate doesn't mean it will be good on the video." for about 2 days or so. It was great, I'm planning to go back next summer.  

My final paramaters for everything after many tweakings are as follows:

* hogcolorspace = 'YCrCb'
* spatialbinsize = (32,32)
* histbinsize = 32
* hog channel = 'ALL'
* hog orient = 9
* hog pixpercell = 8
* hog cellperblock =  2
* window ystart = 400
* window ystop = 650
* window scalestart = 1.4
* window scalestop = 2.0
* color conversion for incoming images  = 'RGB2YCrCb'
* heat threshold - anywhere from 0.5 - 2 seems to be the sweet spot for most images.

This is from code cell 2 of the IPython notebook.

Briefly I tried using the crowdai dataset to augment my training set, but this proved to be unhelpful. The amount of car images then vastly outweighed the non-car image set and created a class bias where everything was identified as a car. I would like to explore this in the future, but my classifier is marginally already good enough at identification for the time being. The extraction of those images can be found in the IPython notebook titled DataExplore.ipynb located [Here](https://github.com/DavidG1011/Udacity---Vehicle-Detection-and-Tracking/blob/master/DataExplore.ipynb)

---

**Video Implementation:**

[link to my video result](./OutputVidFinal.mp4)

The function for the video pipeline is ```videopipeline()``` in code cell 4.

This function simply combines all the previously mentioned sequential steps and functions from code cell 4 but uses frames from the video instead of singular test images as shown before. The ```videopipeline``` function includes the heat mapping implementation to eliminate outliers and focus on strong detections of probable vehicles. I think the pipeline does a reasonable job of detecting the cars in frame. Sometimes cars from the other side of the highway are detected, but they could easily be filtered out if proven troublesome. 

---

**Discussion:**

* The main issues I had with the implementation were mostly optimization related or paramater related. Seemingly, every time I would find a reasonable value for a certain feature extraction on a single frame, my method would be crushed when it came to video output. I would often have issues with false positives or the bounding boxes maybe not quite covering the majority of the car. It was hard to find a good balance of so many different feature extractions where they would work together well.

* Optimization wise, the single frame test images would be output quickly or reasonably fast, but when applied to a full 1200~ frames of a video it proved to slog and take forever to calculate that many frames. During the early testing stages, my video processing time creeped up close to 1 hour and 30 minutes, due to the amount of times I ran a multi-scaled ```find_cars()``` operation. It seemed that every time I would get an accurate video stream, my efficiency for computations would drop entirely. My current video time is a little under 20 minutes to fully render. I am not entirely happy with this, but if I were to speed up the process, I'm afraid I may have to sacrifice accuracy at my current experience level.

* Pipeline Failure: I think that maybe it wouldn't be good as a general "all purpose" classifier. Many of the classification pictures are taken from the project video itself, and while the classifier is good at generalizing, it could maybe still pick up input specific features due to the relatively low amount of training data. As discuseed above in the optimization section-- not the one directly above, I tried to use the crowdai dataset, but I was getting bad results. If I needed to use this pipeline for other videos I would try to extract cars from the crowdai list and use any other location in the frame as a non-car image to balance out the dataset and prevent class bias.
