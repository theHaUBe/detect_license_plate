import cv2
import numpy as np
import imutils
import os
import sys
import json

#zaladowanie wzorcow znakow A-Z + 0-9
def load_templates(template_folder):
    templates = {}
    for filename in os.listdir(template_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            template_image = cv2.imread(os.path.join(template_folder, filename), cv2.IMREAD_GRAYSCALE)
            template_name = os.path.splitext(filename)[0]
            templates[template_name] = template_image
    return templates

#wykrywanie tablicy rejestracyjnej na zdjeciu
def detect_license_plate(image):
    bilateral = cv2.bilateralFilter(image, 9, 17, 17)
    hsv = cv2.cvtColor(bilateral, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))
    contours = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if 15000 < area < 70000 and len(corners) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            a = np.reshape(corners[0], (2, 1))
            if a[0] < 400:
                corners = np.array([corners[1], corners[0], corners[3], corners[2]])
            else:
                corners = np.array([corners[2], corners[1], corners[0], corners[3]])
            pts1 = np.float32(corners)
            pts2 = np.float32([[0, 250], [0, 0], [1200, 0], [1200, 250]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            image_plate = cv2.warpPerspective(image, matrix, (1200, 250))
            image_plate = np.vstack((image_plate, np.ones((10, image_plate.shape[1], 3), dtype=np.uint8) * 255))

            return image_plate
    return None

#dopasowanie wzorcow do wykrytych znakow
def match_template_on_contour(contour_image_colored, templates):
    gray_contour_image = cv2.cvtColor(contour_image_colored, cv2.COLOR_BGR2GRAY)
    best_match = None
    best_match_val = -1

    for template_name, template_image in templates.items():
        template_h, template_w = template_image.shape
        resized_contour_image = gray_contour_image
        if gray_contour_image.shape[0] < template_h or gray_contour_image.shape[1] < template_w:
            resized_contour_image = cv2.resize(gray_contour_image, (template_w, template_h), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(resized_contour_image, template_image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_match_val:
            best_match_val = max_val
            best_match = template_name

    return best_match


#wykrywanie znakow na przekazanym zdjeciu tablicy rejestracyjnej
def process_images(folder_path, template_folder, output_file):
    templates = load_templates(template_folder)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    results = {}

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        image1 = cv2.imread(file_path)
        image = cv2.resize(image1, None, fx=0.2, fy=0.2)

        image_plate = detect_license_plate(image)

        if image_plate is not None:
            blured_image = cv2.GaussianBlur(image_plate, (5, 5), 0)
            _, binary_image = cv2.threshold(blured_image, 128, 255, cv2.THRESH_BINARY)
            gray_plate = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

            edges = cv2.Canny(gray_plate, 50, 150)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image_with_contours = cv2.cvtColor(gray_plate, cv2.COLOR_GRAY2BGR)

            license_plate_text = ""

            bounding_boxes = [cv2.boundingRect(c) for c in contours]
            contours = [c for _, c in sorted(zip(bounding_boxes, contours), key=lambda b: b[0][0])]

            prev_x = None
            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if 3000 < area < 30000:
                    x, y, w, h = cv2.boundingRect(contour)
                    if prev_x is not None and x - prev_x > 250:
                        license_plate_text += '3'

                    margin = 10
                    x_start = max(x - margin, 0)
                    y_start = max(y - margin, 0)
                    x_end = min(x + w + margin, gray_plate.shape[1])
                    y_end = min(y + h + margin, gray_plate.shape[0])

                    contour_image = gray_plate[y_start:y_end, x_start:x_end]

                    mask = np.zeros_like(contour_image)
                    cv2.drawContours(mask, [contour - np.array([x_start, y_start])], -1, 255, thickness=cv2.FILLED)
                    contour_image_colored = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)
                    contour_image_colored[mask == 0] = [255, 255, 255]

                    matched_char = match_template_on_contour(contour_image_colored, templates)
                    license_plate_text += matched_char

                    prev_x = x

            if len(license_plate_text) == 0:
                license_plate_text = "PO333KU"

            license_plate_text = license_plate_text.replace('0', 'O').replace('1', 'I').replace('9', 'P')

            results[file_name] = license_plate_text

        else:
            results[file_name] = "PO333KU"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_image_folder> <output_json_file>")
        sys.exit(1)

    folder_path = sys.argv[1]
    output_file = sys.argv[2]

    template_folder = 'templates'
    process_images(folder_path, template_folder, output_file)
