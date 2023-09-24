<<<<<<< HEAD
from ultralytics import YOLO
import os
import cv2
import csv

def train():
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')
    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='config.yaml', epochs=1000)
    # Evaluate the model's performance on the validation set
    results = model.val()
    # Perform object detection on an image using the model
    results = model('test.jpg')
    print(results)
    # Export the model to ONNX format
    success = model.export(format='onnx')


def recognize(filename):
    VIDEOS_DIR = os.path.join('.', 'videos')
    video_path = os.path.join(VIDEOS_DIR, filename)
    video_path_out = '{}_out.mp4'.format(video_path)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    model_path = os.path.join('.', 'runs', 'detect', 'train10', 'weights', 'last.pt')

    # Load a model
    model = YOLO(model_path)  # load a custom model

    threshold = 0.5
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("--- fps --- {}".format(fps))
    cases = 0
    cur_frame = 0
    cur_min = 0
    cur_sec = 0

    timestamps = list()
    last_frame_detected = 0

    while ret:
        # work with time
        cur_frame += 1
        if(cur_frame >=fps):
            cur_sec += 1
        if(cur_sec > 59):
            cur_min += 1
            cur_sec = 0

        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if (score > threshold) & (cur_frame - last_frame_detected > 3*fps):
                # Detected
                last_frame_detected = cur_frame
                cases += 1
                timestamps.append("{}:{}".format(cur_min, cur_sec))

                print("---------------------------------------------------------------------------------")
                print("frame {} ==== time {}:{} ====".format(cur_frame, cur_min, cur_sec))
                print(result)
                print("---------------------------------------------------------------------------------")
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        out.write(frame)
        ret, frame = cap.read()

    # decision format
    decision = list()
    decision.append(filename)
    decision.append(cases)
    decision.append(timestamps)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return decision

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #train()
    submission = "submission.csv"

    decision = recognize('1.mp4')
    with open(submission, mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(["filename", "cases_count", "timestamps"])
        file_writer.writerow(decision)
=======
from ultralytics import YOLO
import os
import cv2
import csv

def train():
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')
    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='config.yaml', epochs=1000)
    # Evaluate the model's performance on the validation set
    results = model.val()
    # Perform object detection on an image using the model
    results = model('test.jpg')
    print(results)
    # Export the model to ONNX format
    success = model.export(format='onnx')


def recognize(filename):
    VIDEOS_DIR = os.path.join('.', 'videos')
    video_path = os.path.join(VIDEOS_DIR, filename)
    video_path_out = '{}_out.mp4'.format(video_path)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    model_path = os.path.join('.', 'runs', 'detect', 'train10', 'weights', 'last.pt')

    # Load a model
    model = YOLO(model_path)  # load a custom model

    threshold = 0.5
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("--- fps --- {}".format(fps))
    cases = 0
    cur_frame = 0
    cur_min = 0
    cur_sec = 0

    timestamps = list()
    last_frame_detected = 0

    while ret:
        # work with time
        cur_frame += 1
        if(cur_frame >=fps):
            cur_sec += 1
        if(cur_sec > 59):
            cur_min += 1
            cur_sec = 0

        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if (score > threshold) & (cur_frame - last_frame_detected > 180):
                # Detected
                last_frame_detected = cur_frame
                cases += 1
                timestamps.append("{}:{}".format(cur_min, cur_sec))

                print("---------------------------------------------------------------------------------")
                print("frame {} ==== time {}:{} ====".format(cur_frame, cur_min, cur_sec))
                print(result)
                print("---------------------------------------------------------------------------------")
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        out.write(frame)
        ret, frame = cap.read()

    # decision format
    decision = list()
    decision.append(filename)
    decision.append(cases)
    decision.append(timestamps)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return decision

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #train()
    submission = "submission.csv"

    decision = recognize('1.mp4')
    with open(submission, mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(["filename", "cases_count", "timestamps"])
        file_writer.writerow(decision)
>>>>>>> origin/master
