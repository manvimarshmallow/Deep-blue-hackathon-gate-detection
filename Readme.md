Check out the output of the code in the output images folder

*Introduction:*
The problem includes detecting and labeling an underwater gate from a provided dataset by developing a raw image processing algorithm without using ML or YOLO. The result can hence be utilized in the movement of AUV to go through the gate without colliding. The center coordinates of the gate will be useful to the navigation layer of AUV for passing through the gate.

*Approach and Explanation*:
The major techniques employed include :
1) Image Enhancement/Noise Reduction: Histogram Equalization (CLAHE), Gamma Correction, White Balancing, Bilateral Filtering (Gaussian Blur while preserving edges)
2) Edge Detection: Canny Edge Detector with manual thresholds
3) Contour Detection & Filtering: Filtering using the aspect ratio of the bars of the
gate
4) Gate Line Detection: Probabilistic Hough Line Transform
5) Polygon Approximation: Approximation of polygon formed by lines
6) Bounding Box Creation: Creating a bounding box around the region containing
the gate using heuristics involving distance thresholds.
      Below Libraries Used:
is the brief of libraries and functions used:
➔ glob: Used for file path pattern matching.
➔ os: Used for operating system-related operations.
➔ cv2 (OpenCV): Main library for computer vision tasks. ➔ NumPy (np): Used for numerical operations.

*Functions*:
1. decrease_contrast(image, gamma=0.5)
Decreases the contrast of the input image using a gamma correction.
2. preprocess(src, im_resize=1)
Resize the input image (src) by a factor of im_resize.

 Apply noise reduction, histogram equalization, gamma correction, and adaptive thresholding to enhance image features.
Detect edges using the Canny edge detector.
Detect contours in the edge-detected image.
Filter contours based on certain criteria (using functions is_gate_bar and is_gate_top).
Create a binary image highlighting the filtered contours.
Detect lines using the HoughLinesP function.
Draw lines on a black image.
Draw bounding boxes around detected objects based on lines.
Display intermediate and final results using OpenCV.
3. detect_edges(image)
Applies the Canny edge detector to the input image.
4. detect_contours(edges)
Finds contours in the binary edge image.
5. draw_lines(image, lines)
Draws lines on a black image based on the input lines.
6. detect_lines(edges)
Applies the HoughLinesP algorithm to detect lines in the edge-detected image.
7. is_gate_bar(c) and is_gate_top(c)
Helper functions to filter contours based on whether it is a top part of the gate or the
side portion.
8. filter_contours(contours)
Filters contours using the is_gate_bar and is_gate_top functions. It returns a list of valid contours i.e. the ones representing side and top of gate.

 9. create_contours_image(image, contours) Draws contours on a black image.
10. reduce_noise(image)
Applies a bilateral filter for noise reduction.
11. draw_bounding_box(image, lines, distance_threshold=400, iteration_counter=0) Draws bounding boxes around detected objects based on lines.
Performs filtering based on the distance between coordinates.
Returns the modified image, cropped image, and a flag indicating whether a valid bounding box was found.
12. main()
The entry point of the script.
Iterates through images in a specified folder, applies the preprocessing, and displays the results.

Challenges and Solutions:
1. Thedatasetcontainsimagesfromdifferentdistancesthatcanbecategorizedas far, moderate, and close; hence, it is challenging to find a mask common to all images as the threshold ranges are different for them. (A mask is choosing only those pixels you care about). An approach involving generating masks for detecting the vertical and horizontal bars using pixel values is insufficient to handle all kinds of gates since the “blueness” of an image varies exponentially with distance. A probable approach would involve creating all of these masks for all 3 types of gates (close, moderate, and far).
2. Evenforimageswithinthesamecategory,thethresholdvalueshavean inflexible range that takes into consideration lines that are not part of the gate and are a part of the background. To overcome this and avoid manual tuning, we employed basic image enhancement schemes to improve the contrast between foreground and background using gamma correction and histogram equalization on the blue and green channels, as these are the prominent channels in underwater images.

 3. CannyEdgeDetector(withvaluessomewhattuned)withHoughTransformto detect straight lines, detects erroneous lines as well. A probable solution is to improve image enhancement and apply multiple noise reduction techniques to generate a better mask containing gate edges and gate bars solely.
Performance analysis:
We can visualize that with the heuristics developed; we are able to detect the region containing the gate always with performance varying based on the closeness of the gate.
Gate centers were manually marked and compared with the ones obtained via the algorithm, with a ~20% error in the center coordinates on average.
Conclusion:
Overall, we can conclude that image enhancement involving improving contrast, lighting conditions, and segmenting the foreground and background is a crucial part of the pipeline. The efficiency of edge detection and contour detection is majorly affected by the image generated by the image enhancement code.
Other heuristics that can be developed involve shortening the RoI (Region of Interest) by first doing a loose calculation of the position of the gate and then targeting specifically the cropped part of the image for better detection. Depth information, if possible from a depth camera, would also provide helpful information for the detection of gates.

*References for Future Work:*

1) https://github.com/CXH-Research/Underwater-Image-Enhancement (Underwater Image Enhancement)
2) https://openaccess.thecvf.com/content_CVPR_2019/papers/Akkaynak_Sea-Thru _A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_pap er.pdf (Image Enhancement to remove water background)
    
 