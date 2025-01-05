import cv2
import numpy as np
import os
import pytesseract


def process_images(input_dir, output_dir):
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the total number of images to process
    image_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    total_images = len(image_files)

    if total_images == 0:
        print(f"No images found in '{input_dir}'.")
        return

    processed_images = 0  # Counter for processed images

    # Iterate over all files in the input directory
    for filename in image_files:
        file_path = os.path.join(input_dir, filename)

        # Process images
        process_image(file_path, output_dir, filename)
        processed_images += 1
        print(f"Processed {processed_images}/{total_images} images.")  # Live update


def process_videos(input_dir, output_dir):
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the total number of videos to process
    video_files = [
        f for f in os.listdir(input_dir) if f.lower().endswith((".mp4", ".avi", ".mov"))
    ]
    total_videos = len(video_files)

    if total_videos == 0:
        print(f"No videos found in '{input_dir}'.")
        return

    processed_videos = 0  # Counter for processed videos

    # Iterate over all files in the input directory
    for filename in video_files:
        file_path = os.path.join(input_dir, filename)

        # Process videos
        process_video(file_path, output_dir, filename)
        processed_videos += 1
        print(f"Processed {processed_videos}/{total_videos} videos.")  # Live update


def process_image(image_path, output_dir, filename):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to identify the watermark
    edges = cv2.Canny(gray, 50, 150)  # Adjust thresholds to avoid excessive edges

    # Dilate the edges to make the watermark region thicker
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Inpaint the image using the dilated edges as a mask
    result = cv2.inpaint(image, dilated_edges, 3, cv2.INPAINT_TELEA)

    # Resize the result to 1280x1280 pixels
    result_resized = cv2.resize(result, (1280, 1280), interpolation=cv2.INTER_LINEAR)

    # Save the resized result with high quality
    output_path = os.path.join(output_dir, f"removed_watermark_{filename}")
    cv2.imwrite(output_path, result_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Removed display code
    # cv2.imshow(f"Inpainted Image: {filename}", result_resized)
    # cv2.waitKey(0)


def process_video(video_path, output_dir, filename):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    output_path = os.path.join(output_dir, f"watermark_removed_{filename}")
    fourcc = cv2.VideoWriter_fourcc(
        *"XVID"
    )  # You can change this based on your output format
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply edge detection to identify the watermark
        edges = cv2.Canny(gray, 50, 150)  # Adjust thresholds to avoid excessive edges

        # Dilate the edges to make the watermark region thicker
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # Inpaint the frame using the dilated edges as a mask
        result = cv2.inpaint(frame, dilated_edges, 3, cv2.INPAINT_TELEA)

        # Optional: Use OCR to detect text regions and inpaint them
        # Detect text regions only in the lower half of the frame
        lower_half = gray[int(frame_height / 2) :, :]  # Focus on the lower half
        boxes = pytesseract.image_to_boxes(lower_half)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for b in boxes.splitlines():
            b = b.split(" ")
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            # Adjust y-coordinates to match the original frame
            mask[y + int(frame_height / 2) : h + int(frame_height / 2), x:w] = (
                255  # Create a mask for the text regions
            )

        # Inpaint the frame using the text mask
        result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)

        # Write the processed frame to the output video
        out.write(result)

    # Release video objects
    cap.release()
    out.release()

    # Removed display code
    # cv2.imshow(f"Inpainted Video: {filename}", result)
    # cv2.waitKey(0)


# Usage
input_images_dir = "input/images"  # Path to your images folder
output_images_dir = "output/images"  # Path to your output images folder

input_videos_dir = "input/videos"  # Path to your videos folder
output_videos_dir = "output/videos"  # Path to your output videos folder

# Process images and videos separately
process_images(input_images_dir, output_images_dir)
process_videos(input_videos_dir, output_videos_dir)

cv2.destroyAllWindows()
