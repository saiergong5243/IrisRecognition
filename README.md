# GR5293 Project Iris Recognition

## Group Member: Saier Gong (sg3772); Jiefu Zhou (jz3148)
                   

### Whole Logic:

This project implemented the Iris Recognition Algorithm Personal Identification Based on Iris Texture Analysis by Li Ma et al.

IrisRecognition is the main file. Run main() would run all the algorithm step by step including IrisLocalization,IrisNormalization, ImageEnhancement, Feature Extraction, IrisMatching, and PerformanceEnvaluation to give the result plots.

### 1.IrisLocalization

* Project the image in the vertical and horizontal direction to approximately estimate the center coordinates of the pupi. Since the pupil is generally darker than its surroundings, the coordinates corresponding to the minima of the two projection profiles are considered as the center coordinates of the pupil;
* After the first rough estimate of the center of pupil, we binarize a 120 * 120 region centered at that with a threshold of 64;
* Then we use Hough Transformation to find the boundary of pupil and Iris.

### 2. IrisNormalization

* Counterclockwise unwrap the iris ring to a rectangular block with a fixed size. For each pixel in normalized image, find the value for the corresponding pixels in the original image and fill in the value;
* Use the normalized image and rotate it according to the degrees we specified as the training set.

### 3. ImageEnhancement

* Enhance the lighting corrected image by means of histogram equalization in each 32 * 32 region. Such processing compensates for the nonuniform illumination, as well as improves the contrast of the image.

### 4. FeatureExtraction

* Take the top 48 rows as Region of Interest since this region provides most useful texture information;
* Use 2 even-symmetric Gabor filter to do convolution with the image and get two filtered iamges;
* Extract mean and standard deviationfor each 8 * 8 small block as the feature vector.

### 5. IrisMatching

*  Use all functions before and transfer the image database we have into vector database. For each training image, we get 7 vectors since we do rotation for 7 degrees, and for each test image we get 1 vector;
* Apply Linear Discriminant Analysis (LDA) on our dataset and then calculate the L1,L2, and cosine distance and use the class of minimum	distance as the predicted distance.

### 6. PerformanceEnvaluation
+ Use the features we get from previous functions to calculate accuracy rate under different LDA dimensions and under different similarity measurements.
+ The accuracy rates under LDA dimension $[50, 60, 70, 80, 90, 100, 107]$ are all greater than $85\%$ under cosine similarity, and the $80 and 107$ dimensions have the best performance with over $92\%$ accuracy.
+ The original plots and table can be obtain from the HTML file or PDF we submitted. Here, we just provide the screenshots.
![](p1.png)
![](roc.png)
![](table.png)

## Limitation
* We didn't take the noises such as eyelid and eyelash, which can be removed by canny edge detection and Hough algorithm, into consideration;
* We can try more kernels when doing Feature Extraction;
* We could continue to tune the parameters to get a better accuracy rate;
* A larger dataset could help train a better model.

## Peer Evaluation
+ Saier and Jiefu together learned the paper and discussed the background knowledge needed.
+  Then Saier and Jiefu together did the functions, and Saier wrote most parts of the codes, and Jiefu then helped Saier correct some mistakes and wrote instructions of each function.
+  Jiefu wrote the readme file.