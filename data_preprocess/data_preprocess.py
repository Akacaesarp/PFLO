# -*- coding: utf-8 -*-
import base64
import json
import os
import argparse
import io
import random
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


def json2jpg(jsons_dir, save_dir, ratio=1):
    """
    Extract image data from JSON files and save as JPG files

    Parameters:
    jsons_dir: JSON files directory
    save_dir: Directory to save images
    ratio: Image scaling ratio
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    total_files = os.listdir(jsons_dir)
    print(f"Total JSON files: {len(total_files)}")

    for jsonFile in tqdm(total_files, desc="Extracting images"):
        try:
            print(f"Processing: {jsonFile}")
            jsonPath = os.path.join(jsons_dir, jsonFile)
            with open(jsonPath, 'r', encoding='UTF-8') as f:
                data = json.load(f)

                # Check if imageData exists in the JSON file
                if 'imageData' not in data or data['imageData'] is None:
                    print(f"No imageData found in {jsonFile}, skipping...")
                    continue

                image_name = jsonFile.split('.')[0] + ".JPG"
                jpgPath = os.path.join(save_dir, image_name)

                # Check if the same image file already exists in the target path, if so, skip this file
                if os.path.exists(jpgPath):
                    print(f"Image {image_name} already exists, skipping...")
                    continue

                # Safely decode image data
                try:
                    imageData = base64.b64decode(data['imageData'])
                except Exception as e:
                    print(f"Failed to decode image data in {jsonFile}: {e}")
                    continue

                # Write image file
                with open(jpgPath, "wb") as file:
                    file.write(imageData)

                # Process image
                try:
                    image = Image.open(jpgPath)
                    image = ImageOps.exif_transpose(image)
                    image = image.resize((int(image.size[0] / ratio), int(image.size[1] / ratio)))
                    image.save(jpgPath)

                    # Get JSON file modification time
                    src_mtime = os.path.getmtime(jsonPath)
                    src_atime = os.path.getatime(jsonPath)

                    # Apply JSON file modification time to image file
                    os.utime(jpgPath, (src_atime, src_mtime))
                except Exception as e:
                    print(f"Error processing image {image_name}: {e}")
                    if os.path.exists(jpgPath):
                        os.remove(jpgPath)  # Clean up partial files
        except Exception as e:
            print(f"Error processing {jsonFile}: {e}")


def xyxy2xywh(size, box):
    """
    Convert (xmin, ymin, xmax, ymax) format to (x_center, y_center, width, height) format required by YOLO

    Parameters:
    size: Image size (width, height)
    box: Original bounding box coordinates [xmin, ymin, xmax, ymax]

    Returns:
    Normalized (x_center, y_center, width, height) coordinates
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2 * dw  # Center point coordinates relative to original image
    y = (box[1] + box[3]) / 2 * dh
    w = (box[2] - box[0]) * dw
    h = (box[3] - box[1]) * dh
    return (x, y, w, h)  # All returned values are normalized


def convert_label_json(json_dir, save_dir, ratio=1):
    """
    Convert Labelme JSON annotations to YOLO TXT annotations

    Parameters:
    json_dir: Directory containing JSON annotation files
    save_dir: Directory to save TXT annotation files
    ratio: Coordinate scaling ratio
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for json_file in tqdm(os.listdir(json_dir), desc="Converting annotations"):
        json_name = json_file.split('.')[0]
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as load_f:
            print(f"Processing: {json_path}")

            json_dict = json.load(load_f)
            txt_name = json_file.split('.')[0]
            txt_path = os.path.join(save_dir, txt_name + '.txt')

            # If txt file already exists, skip processing this json file
            if os.path.exists(txt_path):
                print(f"Label {txt_path} already exists, skipping...")
                continue

            # Get image size
            try:
                image_data = base64.b64decode(json_dict['imageData'])
                image = Image.open(io.BytesIO(image_data))

                if 'imageHeight' not in json_dict or 'imageWidth' not in json_dict:
                    print(f"Warning: 'imageHeight' or 'imageWidth' not found in {json_path}. Using image size.")
                    w, h = image.size
                else:
                    h = json_dict['imageHeight'] / ratio
                    w = json_dict['imageWidth'] / ratio
            except Exception as e:
                print(f"Error processing image data in {json_file}: {e}")
                continue

            # Save txt file
            with open(txt_path, 'w') as txt_file:
                for shape_dict in json_dict['shapes']:
                    xmin, ymin, xmax, ymax = 0, 0, 0, 0
                    label = shape_dict['label']
                    label_index = 1 if 'y' in label else 0
                    points = shape_dict['points']

                    for point in points:
                        point = [item / ratio for item in point]
                        if xmin == 0:
                            xmin = point[0]
                            ymin = point[1]
                            xmax = point[0]
                            ymax = point[1]
                        else:
                            if point[0] < xmin: xmin = point[0]
                            if point[1] < ymin: ymin = point[1]
                            if point[0] > xmax: xmax = point[0]
                            if point[1] > ymax: ymax = point[1]

                    box = [float(xmin), float(ymin), float(xmax), float(ymax)]
                    # Convert x1, y1, x2, y2 to x, y, w, h format required by YOLO
                    bbox = xyxy2xywh((w, h), box)
                    points_nor_list = []

                    for point in points:
                        point = [item / ratio for item in point]
                        points_nor_list.append(point[0] / w)
                        points_nor_list.append(point[1] / h)
                        points_nor_list.append(1.0)

                    points_nor_list = list(map(lambda x: str(x), points_nor_list))
                    points_nor_str = ' '.join(points_nor_list)

                    label_str = str(label_index) + ' ' + ' '.join(map(str, bbox)) + ' ' + points_nor_str + '\n'
                    txt_file.writelines(label_str)


def sort_and_interpolate_points(json_dir, save_dir, num_selected_points=10, num_pad=10):
    """
    Sort and interpolate points in JSON files

    Parameters:
    json_dir: Directory containing JSON files
    save_dir: Directory to save processed JSON files
    num_selected_points: Final number of points to select
    num_pad: Number of additional points between each pair of points during interpolation
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    json_list = os.listdir(json_dir)
    random.shuffle(json_list)

    for file in tqdm(json_list, desc="Processing points"):
        print(f"Processing: {file}")
        file_path = os.path.join(json_dir, file)

        # If a JSON file with the same name already exists in the target path, skip this file
        new_file_path = os.path.join(save_dir, file)
        if os.path.exists(new_file_path):
            print(f"Processed file {file} already exists, skipping...")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                for item in data['shapes']:
                    if 'y' not in item['label']:
                        # Process root points
                        root = item['points']
                        sorted_root = sorted(root, key=lambda x: x[1], reverse=True)
                        item['points'] = sorted_root
                    else:
                        belong_root = None
                        belong_label = item['label'].split('-')[0]

                        for temp in data['shapes']:
                            if belong_label == temp['label']:
                                belong_root = temp['points']
                                break

                        if belong_root is not None:
                            mean_x = sum(point[0] for point in belong_root) / len(belong_root)
                            mean_y = sum(point[1] for point in belong_root) / len(belong_root)

                            length_first = (item['points'][0][0] - mean_x) ** 2 + (item['points'][0][1] - mean_y) ** 2
                            length_last = (item['points'][-1][0] - mean_x) ** 2 + (item['points'][-1][1] - mean_y) ** 2

                            if length_first > length_last:
                                item['points'] = item['points'][::-1]

                    # Interpolate to increase point count
                    new_points = []
                    if len(item['points']) < 2:
                        print(f"Warning: {file}'s {item['label']} has fewer than 2 points, cannot interpolate. Will use original points or default points.")

                        # If original points are not empty, use original points for filling
                        if item['points']:
                            new_points = item['points'] * num_selected_points
                        else:
                            # If there are no points at all, fill with default points
                            new_points = [[0, 0]] * num_selected_points
                    else:
                        for i in range(len(item['points']) - 1):
                            x1, y1 = item['points'][i]
                            x2, y2 = item['points'][i + 1]

                            for j in range(num_pad + 1):
                                x = x1 + (x2 - x1) * j / (num_pad + 1)
                                y = y1 + (y2 - y1) * j / (num_pad + 1)
                                new_points.append([x, y])

                            if i == len(item['points']) - 2:
                                new_points.append([x2, y2])

                    # Ensure new_points count is not less than num_selected_points
                    if len(new_points) < num_selected_points:
                        print(
                            f"Warning: {file}'s {item['label']} has fewer than {num_selected_points} interpolated points, currently {len(new_points)}. Will repeat the last point to fill.")

                        # If new_points is empty, fill with default points
                        if not new_points:
                            new_points = [[0, 0]] * num_selected_points
                        else:
                            while len(new_points) < num_selected_points:
                                new_points.append(new_points[-1])  # Repeat the last point to fill

                    # Calculate step size, ensure the last point can also be selected
                    num_points = len(new_points)
                    step = (num_points - 1) / (num_selected_points - 1) if num_selected_points > 1 else 0

                    # Select fixed number of points
                    selected_points = []
                    for i in range(num_selected_points):
                        index = round(i * step)  # Use round instead of int to reduce error
                        if index >= len(new_points):
                            index = len(new_points) - 1  # Prevent out of bounds
                        selected_points.append(new_points[index])

                    # Check point count
                    if len(selected_points) != num_selected_points:
                        print(
                            f"Warning: {file}'s {item['label']} has {len(selected_points)} target points, should be {num_selected_points}")

                    item['points'] = selected_points

            # Save new JSON file
            with open(new_file_path, 'w', encoding='utf-8') as new_f:
                json.dump(data, new_f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error processing {file}: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Data Preprocessing Tool')
    parser.add_argument('--task', type=str, required=True, choices=['extract', 'convert', 'process', 'all'],
                        help='Task to execute: extract (extract images), convert (convert annotations), process (process points), all (all tasks)')
    parser.add_argument('--json-dir', type=str, required=True,
                        help='JSON files directory')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Save directory')
    parser.add_argument('--extract-dir', type=str,
                        help='Directory to save extracted images (only needed when task=all)')
    parser.add_argument('--convert-dir', type=str,
                        help='Directory to save converted annotations (only needed when task=all)')
    parser.add_argument('--process-dir', type=str,
                        help='Directory to save processed points (only needed when task=all)')
    parser.add_argument('--ratio', type=float, default=1,
                        help='Scaling ratio (default: 1)')
    parser.add_argument('--num-points', type=int, default=10,
                        help='Number of points to select (default: 10)')
    parser.add_argument('--num-pad', type=int, default=10,
                        help='Number of additional points between each pair of points during interpolation (default: 10)')
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    if args.task == 'extract' or args.task == 'all':
        save_dir = args.save_dir if args.task == 'extract' else args.extract_dir
        print(f"\n=== Starting to extract images from JSON to {save_dir} ===")
        json2jpg(args.json_dir, save_dir, args.ratio)

    if args.task == 'convert' or args.task == 'all':
        save_dir = args.save_dir if args.task == 'convert' else args.convert_dir
        print(f"\n=== Starting to convert JSON annotations to YOLO format and save to {save_dir} ===")
        convert_label_json(args.json_dir, save_dir, args.ratio)

    if args.task == 'process' or args.task == 'all':
        save_dir = args.save_dir if args.task == 'process' else args.process_dir
        print(f"\n=== Starting to process points and save to {save_dir} ===")
        sort_and_interpolate_points(args.json_dir, save_dir, args.num_points, args.num_pad)

    print("\nAll tasks completed!")


if __name__ == "__main__":
    main()