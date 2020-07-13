"""
Egestor for darknet formats.
"""

import os
import shutil

from data.converter.converter import Egestor

class DarknetEgoEgestor(Egestor):
    # These labels are set specifically for egohands
    def expected_labels(self):
        return {
            'operatorleft': [],
            'operatorright': [],
            'handleft': [],
            'handright': []
        }

    def egest(self, *, image_detections, root):
        # Create paths to hold images and associated labels
        images_path = f"{root}images"
        labels_path = f"{root}labels"
        # Create directories
        for to_create in [images_path, labels_path]:
            os.makedirs(to_create, exist_ok=True)
        # Loop through all images
        for image_detection in image_detections:
            # Get image schema and use to copy image to directory path
            image = image_detection['image']
            image_id = image['id']
            src_extension = image['path'].split('.')[-1]
            shutil.copyfile(image['path'], f"{images_path}/{image_id}.{src_extension}")
            # Get image height and width
            img_height = image['height']
            img_width = image['width']
            # Create an array to hold all detected object strings
            objects = []
            # For each image, there exist a number of possible detections associated with it (tied to classes)
            # Loop through all iage detections and add as a single row for each
            for detection in image_detection['detections']:
                # Each detection should have a label, and values for (top, bottom, left and right) xy coords
                # We need to create a single row string for each detection with the following structure:
                # class, x_center, y_center, width, height
                # Le Sigh!, therefore we must convert them so they match up
                bb_top = detection['top']
                bb_bottom = detection['bottom']
                bb_left = detection['left']
                bb_right = detection['right']
                # set bb height and width
                bb_width =  bb_right - bb_left
                bb_height = bb_bottom - bb_top
                # Set x and y centre's
                x_centre = bb_left + ((bb_right - bb_left) / 2)
                y_centre = bb_top + ((bb_bottom - bb_top) / 2)
                # Next normalise values by dividing x_center and width by image width, and y_center and height by image height.
                x_centre = x_centre / img_width
                y_centre = y_centre / img_height
                bb_width = bb_width / img_width
                bb_height = bb_height / img_height
                # Check negatives
                if x_centre or y_centre or bb_width or bb_height < 0:
                    assert "error"
                # Get label
                label = detection['label']
                # Convert label to darknet zeroindex format
                # Track which hand is being enumerated
                if label == "operatorleft":
                    label = 0
                elif label == "operatorright":
                    label = 1
                elif label == "handleft":
                    label = 2
                elif label == "handright":
                    label = 3
                # Create row string
                object = str(label) + " " + "{:.6f}".format(x_centre) + " " + "{:.6f}".format(y_centre) + " " + "{:.6f}".format(bb_width) + " " + "{:.6f}".format(bb_height)
                # Add to list
                objects.append(object)

            # Write labels file for this image
            file = open(f"{labels_path}/{image_id}" + ".txt", "w+")
            # Loop through each object and add to labels file
            for object in objects:
                # Add
                file.write(object + "\n")
            # Close file
            file.close()













