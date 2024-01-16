from glob import glob
import os
import cv2
import numpy as np

def decrease_contrast(image, gamma=0.5):
    gamma_corrected = np.power(image / 255.0, gamma) * 255.0
    gamma_corrected = np.uint8(gamma_corrected)
    return gamma_corrected

def preprocess(src, im_resize=1):


    im_dims = (int(src.shape[1]*im_resize), int(src.shape[0]*im_resize))
    if im_resize != 1.0:
        split = cv2.split(src)
        kernel = (3,3)
        sig = 1
        temp = [reduce_noise(channel) for channel in split]
        src = cv2.merge(temp)
        src = cv2.resize(src, im_dims, cv2.INTER_CUBIC)
    cv2.imshow("Original", src)
    b,g,r = cv2.split(src)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b_enhanced = clahe.apply(b)
    g_enhanced = clahe.apply(g)
    merged_image = cv2.merge([b_enhanced,g_enhanced,r])
    gray_image = cv2.cvtColor(merged_image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    equalized_image = reduce_noise(equalized_image)
    range_image = np.uint8(np.minimum(np.uint32(equalized_image) + 10, 255))
    gamma_image = decrease_contrast(range_image, 0.1)
    gamma_image = np.float32(gamma_image)
    gamma_image = ((gamma_image-np.amin(gamma_image))/(np.amax(gamma_image)-np.amin(gamma_image)))*255
    gamma_image = np.uint8(gamma_image)
    gamma_image = reduce_noise(gamma_image)
    # gamma_image = cv2.GaussianBlur(gamma_image,(3,3),1.0)
    block_size = 11  
    constant = 5 
    adaptive_thresholded_image = cv2.adaptiveThreshold(
        gamma_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, constant
    )
    edges = detect_edges(gamma_image)
    # cv2.imshow("Test_Equalized", equalized_image)
    # cv2.imshow("Test_Gamma", gamma_image)
    # cv2.imshow("Test_Adapt", adaptive_thresholded_image)

    contours = detect_contours(edges)
    filtered = filter_contours(contours)
    contours_image = create_contours_image(edges.copy(), filtered)

    lines = detect_lines(contours_image)
    draw_lines(contours_image, lines)
    final, cropped,  _ = draw_bounding_box(src, lines)
    # cv2.imshow("Edges", edges)
    # cv2.imshow("Filtered", contours_image)
    cv2.imshow("Final", final)
    # cv2.imshow("Cropped", cropped)
    # cv2.waitKey(0)
    return cropped, _

def detect_edges(image):
    edges = cv2.Canny(image, 80, 150)
    return edges

def detect_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_lines(image, lines):
    image = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.imshow("Lines", image)
    return image

def detect_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=8) 
    return lines

def is_gate_bar(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True) 
    (x, y, w, h) = cv2.boundingRect(approx) 

    if h < (3.5*w): 
        return False, (x, y, w, h)
    return True, (x, y, w, h)

def is_gate_top(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True) 
    (x, y, w, h) = cv2.boundingRect(approx) 
    if w < h: 
        return False, (x, y, w, h)
    return True, (x, y, w, h)

def filter_contours(contours):
    filtered_contours = [c for c in contours if is_gate_bar(c)[0]]
    top_contours = [c for c in contours if is_gate_top(c)[0]]
    for c in top_contours:
        filtered_contours.append(c)
    return filtered_contours

def create_contours_image(image, contours):
    contours_image = np.zeros_like(image)
    cv2.drawContours(contours_image, contours, -1, (255, 255, 255), thickness=2)
    return contours_image


def reduce_noise(image):
    # Apply bilateral filter for noise reduction
    denoised_image = cv2.bilateralFilter(image, d=9, sigmaColor=20, sigmaSpace=3)
    return denoised_image

def draw_bounding_box(image, lines, distance_threshold=400, iteration_counter=0):
    if lines is not None and len(lines) > 0:
        coordinates = np.array([line[0] for line in lines])
        if len(coordinates) > 0:
            try:
                distances = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
                valid_indices = np.where(distances < distance_threshold)[0]
                filtered_coordinates = coordinates[valid_indices]      
                if len(filtered_coordinates) > 0:
                    x_min, y_min = np.min(filtered_coordinates, axis=0)[:2]
                    x_max, y_max = np.max(filtered_coordinates, axis=0)[2:]       
                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2

                    bounding_box_area = (x_max - x_min) * (y_max - y_min)
                    image_area = image.shape[0] * image.shape[1] // 2
                    cropped_image = np.zeros_like(image)
                    pos = False
                    if bounding_box_area > image_area:
                        cropped_image = image[y_min:y_max, x_min:x_max]
                        cropped_image = cropped_image.copy()
                        pos = True
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
            except cv2.error as e:
                print("Error in drawing bounding box:", e)
    return image, cropped_image, pos

def main():

    folder_path = '/Users/manvibengani/Desktop/Deep-blue-hackathon-gate-detection/dataset'

    for path in os.listdir(folder_path):#loop to read one image at a time 
        image = os.path.join(folder_path, path)
        img, more = preprocess(cv2.imread(image),0.4)
        # if more:
        #     img,_ = preprocess(img,1.0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

main()