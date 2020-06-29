from models.mobilenetV1_SSD import create_mobilenetv1_ssd_predictor
from models.mobilenetV1_SSD import MobileNetV1_SSD
from utils.misc import Timer
import cv2
import sys

"""
#######################
Description
#######################

This script is used to demonstrate live the trained model in object detection and localisation. You can essentially pass
it a video file or leave empty to use the webcam.

Example input to run script: live_run.py
# Then include the following below, if you leave out video path, it goes to webcam, also argument 4 is for flipping image
# data/egohands/models/mobilenetv1-SSD-Epoch-140-Loss-1.4216.pth
# data/egohands/egohandsVOC/ego-model-labels.txt
# examples/videos/Bending.mp4
# True

i.e live_run.py data/egohands/models/mobilenetv1-SSD-Epoch-140-Loss-1.4216.pth data/egohands/egohandsVOC/ego-model-labels.txt examples/videos/Bending.mp4 True

More example videos can be found in 'examples/' they consist of some videos from the egohands dataset and some videos from
the cable manipulation dataset (personal). For the cable manipulation examples, set flip = True so they better align with egohands orientation 

"""
# Run either the SSD model agasint a supplied video file, or if not present live through a webcam
if len(sys.argv) < 4:
    print('Usage: python live_run.py <model path> <label path> [video file]')
    sys.exit(0)

model_path = sys.argv[1]
label_path = sys.argv[2]
flip_image = sys.argv[4]

if len(sys.argv) >= 4:
    cap = cv2.VideoCapture(sys.argv[3])  # capture from file
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

# Create our mobilenetv1 model
net = MobileNetV1_SSD(num_classes=len(class_names), is_test=True)
net.load(model_path)

# Create predictor
predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

timer = Timer()
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    # Check whether to flip image orientation horizontally
    if (flip_image == "True" or flip_image == "true"):
        # Flip image
        orig_image = cv2.flip(orig_image, 0)
    # Convert to correct colour range
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()