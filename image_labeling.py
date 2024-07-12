import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_raw_image(image_path):
    """Load the raw image from the specified path."""
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise ValueError("Image not found or invalid image format.")
    return raw_image

def preprocess_image(image):
    """Convert the image to grayscale."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def load_labeled_image(label_image_path):
    """Load the labeled image from the specified path."""
    labeled_image = cv2.imread(label_image_path, cv2.IMREAD_COLOR)
    if labeled_image is None:
        raise ValueError("Labeled image not found or invalid format.")
    return labeled_image

def save_labeled_image(image, output_path):
    """Save the labeled image to the specified path."""
    cv2.imwrite(output_path, image)

def main():
    """Main function to execute the image loading, preprocessing, and displaying."""
    raw_image_path = 'labeled-forest-scene-svg.png'  # Replace with your raw image path
    label_image_path = 'labeled-forest-scene-svg copy.png'  # Replace with your labeled image path
    output_path = 'output_labeled_image.png'  # Replace with your output path

    try:
        raw_image = load_raw_image(raw_image_path)
        preprocessed_image = preprocess_image(raw_image)
        labeled_image = load_labeled_image(label_image_path)
        save_labeled_image(labeled_image, output_path)
    except ValueError as e:
        print(e)
        return

    # Display images using matplotlib
    plt.subplot(1, 2, 1)
    plt.title('Raw Image')
    plt.imshow(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title('Labeled Image')
    plt.imshow(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB))

    plt.show()

if __name__ == "__main__":
    main()

    # Interactive labeling
    raw_image_path = 'labeled-forest-scene-svg.png'  # Replace with your raw image path
    raw_image = load_raw_image(raw_image_path)
    preprocessed_image = preprocess_image(raw_image)
    labeled_image = np.zeros_like(raw_image, dtype=np.uint8)

    drawing = False
    ix, iy = -1, -1

    def draw_label(event, x, y, flags, param):
        global ix, iy, drawing, labeled_image

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(labeled_image, (x, y), 5, (0, 0, 255), -1)  # Draw in red color
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(labeled_image, (x, y), 5, (0, 0, 255), -1)  # Draw in red color

    cv2.namedWindow('Label Image')
    cv2.setMouseCallback('Label Image', draw_label)

    while True:
        combined_image = cv2.addWeighted(raw_image, 0.7, labeled_image, 0.3, 0)
        cv2.imshow('Label Image', combined_image)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Esc' key
            break

    cv2.destroyAllWindows()
    save_labeled_image(labeled_image, 'interactive_labeled_image.png')

