"""
Ingestor and egestor for egohands formats.
http://vision.soic.indiana.edu/projects/egohands/
"""

import os
import scipy.io as sio
import cv2
import numpy as np
import re as re

from data.converter.converter import Ingestor

class EgoHandsIngestor(Ingestor):
    def validate(self, path):
        # Validate that the format of the dataset / images is correct
        for subdir in ["CARDS_COURTYARD_B_T", "CARDS_COURTYARD_H_S", "CARDS_COURTYARD_S_H",
                       "CARDS_COURTYARD_T_B", "CARDS_LIVINGROOM_B_T", "CARDS_LIVINGROOM_H_S",
                       "CARDS_LIVINGROOM_S_H", "CARDS_LIVINGROOM_T_B",
                       "CARDS_OFFICE_B_S", "CARDS_OFFICE_H_T", "CARDS_OFFICE_S_B",
                       "CARDS_OFFICE_T_H", "CHESS_COURTYARD_B_T", "CHESS_COURTYARD_H_S",
                       "CHESS_COURTYARD_S_H", "CHESS_COURTYARD_T_B", "CHESS_LIVINGROOM_B_S",
                       "CHESS_LIVINGROOM_H_T", "CHESS_LIVINGROOM_S_B", "CHESS_LIVINGROOM_T_H",
                       "CHESS_OFFICE_B_S", "CHESS_OFFICE_H_T", "CHESS_OFFICE_S_B", "CHESS_OFFICE_T_H",
                       "JENGA_COURTYARD_B_H", "JENGA_COURTYARD_H_B", "JENGA_COURTYARD_S_T",
                       "JENGA_COURTYARD_T_S", "JENGA_LIVINGROOM_B_H", "JENGA_LIVINGROOM_H_B",
                       "JENGA_LIVINGROOM_S_T", "JENGA_LIVINGROOM_T_S", "JENGA_OFFICE_B_S",
                       "JENGA_OFFICE_H_T", "JENGA_OFFICE_S_B", "JENGA_OFFICE_T_H",
                       "PUZZLE_COURTYARD_B_S", "PUZZLE_COURTYARD_H_T", "PUZZLE_COURTYARD_S_B",
                       "PUZZLE_COURTYARD_T_H", "PUZZLE_LIVINGROOM_B_T", "PUZZLE_LIVINGROOM_H_S",
                       "PUZZLE_LIVINGROOM_S_H", "PUZZLE_LIVINGROOM_T_B", "PUZZLE_OFFICE_B_H",
                       "PUZZLE_OFFICE_H_B", "PUZZLE_OFFICE_S_T", "PUZZLE_OFFICE_T_S"]:
            if not os.path.isdir(f"{path}/{subdir}"):
                return False, f"Expected subdirectory {subdir} within {path}"
            if not os.path.isfile(f"{path}/{subdir}/polygons.mat"):
                return False, f"Expected polygons.mat in folder {path}"
        return True, None

    def preprocessing(self, path=""):
        # Perform some pre-processing on egohands dataset in order to rename image ids to associated sets
        # Loop through all image set subdirectories and rename files to include set name if not already present
        for subdir in ["CARDS_COURTYARD_B_T", "CARDS_COURTYARD_H_S", "CARDS_COURTYARD_S_H",
                       "CARDS_COURTYARD_T_B", "CARDS_LIVINGROOM_B_T", "CARDS_LIVINGROOM_H_S",
                       "CARDS_LIVINGROOM_S_H", "CARDS_LIVINGROOM_T_B",
                       "CARDS_OFFICE_B_S", "CARDS_OFFICE_H_T", "CARDS_OFFICE_S_B",
                       "CARDS_OFFICE_T_H", "CHESS_COURTYARD_B_T", "CHESS_COURTYARD_H_S",
                       "CHESS_COURTYARD_S_H", "CHESS_COURTYARD_T_B", "CHESS_LIVINGROOM_B_S",
                       "CHESS_LIVINGROOM_H_T", "CHESS_LIVINGROOM_S_B", "CHESS_LIVINGROOM_T_H",
                       "CHESS_OFFICE_B_S", "CHESS_OFFICE_H_T", "CHESS_OFFICE_S_B", "CHESS_OFFICE_T_H",
                       "JENGA_COURTYARD_B_H", "JENGA_COURTYARD_H_B", "JENGA_COURTYARD_S_T",
                       "JENGA_COURTYARD_T_S", "JENGA_LIVINGROOM_B_H", "JENGA_LIVINGROOM_H_B",
                       "JENGA_LIVINGROOM_S_T", "JENGA_LIVINGROOM_T_S", "JENGA_OFFICE_B_S",
                       "JENGA_OFFICE_H_T", "JENGA_OFFICE_S_B", "JENGA_OFFICE_T_H",
                       "PUZZLE_COURTYARD_B_S", "PUZZLE_COURTYARD_H_T", "PUZZLE_COURTYARD_S_B",
                       "PUZZLE_COURTYARD_T_H", "PUZZLE_LIVINGROOM_B_T", "PUZZLE_LIVINGROOM_H_S",
                       "PUZZLE_LIVINGROOM_S_H", "PUZZLE_LIVINGROOM_T_B", "PUZZLE_OFFICE_B_H",
                       "PUZZLE_OFFICE_H_B", "PUZZLE_OFFICE_S_T", "PUZZLE_OFFICE_T_S"]:
            # Loop through all image in subdirectory
            for file in os.listdir(path + subdir):
                if file.endswith(".jpg"):
                    # Check whether we need to rename
                    if file.__contains__(subdir):
                        continue
                    else:
                        # Rename file to include subdirectory of image set as header
                        os.rename(path + subdir + "/" + file, path + subdir + "/" + subdir + "_" + file)

    def ingest(self, path):
        # Using path "/egohands_data/_LABELLED_SAMPLES/" which contains subdirectories of image sets
        # Perform any preprocessing of dataset
        self.preprocessing(path=path)
        # Construct an array holding each images information and detected objects
        image_objects = self.createImageDetectionArray(path=path)
        # Return
        return image_objects

    def createImageDetectionArray(self, path=""):
        # Egohands is built into separate folders of unique sets of images and associated detected objects:
        # (myhandLeft, myhandRight, yourhandLeft, yourhandRight). The bbox data is essentially held as list of data points
        # associated with the hand segmentation, and therefore we need to enumerate these data points to get min/max for bbox
        # For each image we need to construct an object array:
        """"
        {
            'image': {
                'id': image_id,
                'path': image_path,
                'segmented_path': segmented_path,
                'width': image_width,
                'height': image_height
            },
            'detections': [self._get_detection(node) for node in xml_root.findall('object')]
        }

        ********* IF AN IMAGE DOES NOT HAVE AN ASSOCIATED OBJECT/S BOX! THEN IT WILL NOT BE INCLUDED *********

        """

        # Create an array to hold all image objects
        imageObjectArray = []
        # Loop through all image subdirectories
        for subdir in ["CARDS_COURTYARD_B_T", "CARDS_COURTYARD_H_S", "CARDS_COURTYARD_S_H",
                       "CARDS_COURTYARD_T_B", "CARDS_LIVINGROOM_B_T", "CARDS_LIVINGROOM_H_S",
                       "CARDS_LIVINGROOM_S_H", "CARDS_LIVINGROOM_T_B",
                       "CARDS_OFFICE_B_S", "CARDS_OFFICE_H_T", "CARDS_OFFICE_S_B",
                       "CARDS_OFFICE_T_H", "CHESS_COURTYARD_B_T", "CHESS_COURTYARD_H_S",
                       "CHESS_COURTYARD_S_H", "CHESS_COURTYARD_T_B", "CHESS_LIVINGROOM_B_S",
                       "CHESS_LIVINGROOM_H_T", "CHESS_LIVINGROOM_S_B", "CHESS_LIVINGROOM_T_H",
                       "CHESS_OFFICE_B_S", "CHESS_OFFICE_H_T", "CHESS_OFFICE_S_B", "CHESS_OFFICE_T_H",
                       "JENGA_COURTYARD_B_H", "JENGA_COURTYARD_H_B", "JENGA_COURTYARD_S_T",
                       "JENGA_COURTYARD_T_S", "JENGA_LIVINGROOM_B_H", "JENGA_LIVINGROOM_H_B",
                       "JENGA_LIVINGROOM_S_T", "JENGA_LIVINGROOM_T_S", "JENGA_OFFICE_B_S",
                       "JENGA_OFFICE_H_T", "JENGA_OFFICE_S_B", "JENGA_OFFICE_T_H",
                       "PUZZLE_COURTYARD_B_S", "PUZZLE_COURTYARD_H_T", "PUZZLE_COURTYARD_S_B",
                       "PUZZLE_COURTYARD_T_H", "PUZZLE_LIVINGROOM_B_T", "PUZZLE_LIVINGROOM_H_S",
                       "PUZZLE_LIVINGROOM_S_H", "PUZZLE_LIVINGROOM_T_B", "PUZZLE_OFFICE_B_H",
                       "PUZZLE_OFFICE_H_B", "PUZZLE_OFFICE_S_T", "PUZZLE_OFFICE_T_S"]:
            # Loop through all images in subdirectory, using polygons.mat to create object list
            image_path_array = []
            # Store image ids
            image_ids = []
            # Loop through all image in subdirectory
            for file in os.listdir(path + subdir):
                if file.endswith(".jpg"):
                    # Get image path
                    img_path = path + subdir + "/" + file
                    # Get image filename
                    file = re.sub('\.jpg$', '', file)
                    image_ids.append(file)
                    # Get image name and append
                    image_path_array.append(img_path)
            # Sort image paths
            image_path_array.sort()
            # Sort image IDs
            image_ids.sort()
            # Load polygons.mat file for this subdirectory, it has 100 sets of values (objects)
            boxes = sio.loadmat(path + subdir + "/polygons.mat")
            # Get object data points
            polygons = boxes["polygons"][0]
            # Set point index to 0
            pointindex = 0
            # Loop through each object within
            for polygon in polygons:
                # Create our image object array
                image_object = {
                    'image': {
                        'id': "",
                        'path': "",
                        'segmented_path': None,
                        'width': 0,
                        'height': 0
                    },
                    'detections': []
                }
                # Show images as we create object, set font
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Get image ID
                image_id = image_ids[pointindex]
                # Get image path ID
                image_path_id = image_path_array[pointindex]
                # Get image file
                image = cv2.imread(image_path_id)
                # Pass data to image object
                image_object['image']['id'] = image_id
                image_object['image']['path'] = image_path_id
                image_object['image']['segmented_path'] = None
                # Get image shape
                image_width = np.size(image, 1)
                image_height = np.size(image, 0)
                # Pass to image object
                image_object['image']['width'] = image_width
                image_object['image']['height'] = image_height
                # Increment point index
                pointindex += 1
                ### Determine all objects detected within image ###
                # Decalre box array
                detections = []
                # Loop through each detected object in image
                for i, pointlist in enumerate(polygon):
                    # Create object to hold bounding box information
                    bbox = {
                        'label': "",
                        'top': 0,
                        'left': 0,
                        'right': 0,
                        'bottom': 0
                    }
                    # Set hand label depending on index value
                    hand = ""
                    # Track which hand is being enumerated
                    if i == 0:
                        hand = "operatorleft"
                    elif i == 1:
                        hand = "operatorright"
                    elif i == 2:
                        hand = "handleft"
                    elif i == 3:
                        hand = "handright"
                    # Pass to bbox array
                    bbox['label'] = hand
                    # We now have to enumerate through all points in the pointlist. This is a set of points which outlines
                    # the segmentation for the object in the image. Therefore use min and max values to determine bbox
                    # Declare variables to hold min and max for xy
                    pst = np.empty((0, 2), int)
                    max_x = max_y = min_x = min_y = 0
                    findex = 0
                    # Loop through point list
                    for point in pointlist:
                        if (len(point) == 2):
                            x = int(point[0])
                            y = int(point[1])

                            if (findex == 0):
                                min_x = x
                                min_y = y
                            findex += 1
                            max_x = x if (x > max_x) else max_x
                            min_x = x if (x < min_x) else min_x
                            max_y = y if (y > max_y) else max_y
                            min_y = y if (y < min_y) else min_y
                            append = np.array([[x, y]])
                            pst = np.append(pst, append, axis=0)
                            # Show on displayed image object class text
                            cv2.putText(image, ".", (x, y), font, 0.7,
                                        (255, 255, 255), 2, cv2.LINE_AA)
                    # Pass values to object array
                    bbox['top'] = min_y
                    bbox['left'] = min_x
                    bbox['right'] = max_x
                    bbox['bottom'] = max_y
                    # Check whether to add this image object or not
                    if (bbox['top'] <= 0 and bbox['bottom'] <= 0 and bbox['right'] <= 0 and bbox['left'] <= 0 ):
                        # This bounding box does not conform to an actual object, so ignore it.
                        pass
                    else:
                        # Add bounding box object to array
                        detections.append(bbox)
                    # Validate image object has been correctly processed through cv viewing, add box lines and
                    # a connected line highlighting segmentation
                    cv2.polylines(image, [pst], True, (0, 255, 255), 1)
                    cv2.rectangle(image, (min_x, max_y),
                                  (max_x, min_y), (0, 255, 0), 1)

                # Add detections to image object
                image_object['detections'] = detections
                # Check whether to add this image object to array, if it does not have any detections / labels then DO NOT ADD!
                if (detections == []):
                    # Ignore
                    pass
                else:
                    # Add to object array
                    imageObjectArray.append(image_object)
                # Show image
                cv2.putText(image, "DIR : " + subdir + " - " + image_id, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                cv2.imshow('Verifying annotation ', image)
                # Save image for one subdirectory as an example
                #if subdir == "CARDS_COURTYARD_B_T":
                    #cv2.imwrite("exampleImages/" + "DIR : " + subdir + " - " + image_id + ".jpg", image)
                cv2.waitKey(1)  # Change this to 1000 to see every single frame
        # Return
        return imageObjectArray