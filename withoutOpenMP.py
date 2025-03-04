import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# Edge detection function without OpenMP (simplified)
def edge_detection(image):
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.uint8)

    # Sobel operators
    gx = np.array([[-1, 0, 1], 
                   [-2, 0, 2], 
                   [-1, 0, 1]])
    gy = np.array([[-1, -2, -1], 
                   [0, 0, 0], 
                   [1, 2, 1]])

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            sum_x = 0
            sum_y = 0
            for k in range(3):
                for l in range(3):
                    pixel = image[i + k - 1, j + l - 1]
                    sum_x += gx[k][l] * pixel
                    sum_y += gy[k][l] * pixel
            edge_strength = np.sqrt(sum_x**2 + sum_y**2)
            if edge_strength > 255:
                edge_strength = 255
            output[i, j] = edge_strength
    return output

def fingerprint_preprocessing(image):
    # Step 1: Histogram Equalization
    img_eq = cv2.equalizeHist(image)

    # Step 2: Gaussian Blur to remove noise
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)

    # Step 3: Adaptive Thresholding for binary image
    img_thresh = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    return img_thresh

def main():
    # Load the fingerprint image in grayscale
    img = cv2.imread('fingerprint.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: fingerprint.jpg not found.")
        return

    # Preprocess the fingerprint image
    preprocessed_img = fingerprint_preprocessing(img)

    start_time = time.time()
    # Apply edge detection on the preprocessed fingerprint
    edges = edge_detection(preprocessed_img)
    end_time = time.time()

    print(f"Processing time: {end_time - start_time:.4f} seconds")

    # Display images using Matplotlib
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('Original Fingerprint')
    plt.xticks([]), plt.yticks([])

    plt.subplot(132)
    plt.imshow(preprocessed_img, cmap='gray')
    plt.title('Preprocessed Fingerprint')
    plt.xticks([]), plt.yticks([])

    plt.subplot(133)
    plt.imshow(edges, cmap='gray')
    plt.title('Fingerprint Edges')
    plt.xticks([]), plt.yticks([])

    plt.show()

    # Save Edge Detection Image
    cv2.imwrite('fingerprint_edges.jpg', edges)
    print("Fingerprint edge-detected image saved as fingerprint_edges.jpg")

if __name__ == "__main__":
    main()

