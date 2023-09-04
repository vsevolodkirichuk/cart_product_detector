import argparse
import torch
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--labels_path', type=str, default='data/keys_yolo.pt')
parser.add_argument('--task_type', type=str, default='video')
parser.add_argument('--od_model_path', type=str, default='data/best.pt')
args = parser.parse_args()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def center_crop(img, dim):
    """Returns center cropped image
    Args:
    img: image to be center cropped   #
    dim: dimensions (width, height) to be cropped.
    """
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img


def preprocess(img):
    """Returns resized and normalized image
    Args:
    img: image to resized and normalized
    """
    img = cv2.resize(img, (256, 256))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = center_crop(img, (254, 254))
    img = img / 255.0

    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]

    img[..., 0] /= std[0]
    img[..., 1] /= std[1]
    img[..., 2] /= std[2]
    im = img[..., ::-1].transpose((2, 0, 1))
    return torch.Tensor(np.ascontiguousarray(im))

def get_bboxes(bb:list, width:int, height:int)->list:
    bb[0] *= width
    bb[2] *= width
    bb[1] *= height
    bb[3] *= height
    return [int(i) for i in bb[:4]]

def main(args: dict):
    if args.task_type == 'video': cap = cv2.VideoCapture('C:/Deep Learning/cart_product_detector/data/videos/IMG_0991.MOV')
    else: cap = cv2.VideoCapture(0)
    yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=args.od_model_path)
    labels = torch.load(args.labels_path)
    #print(labels)
    while (cap.isOpened()):
        print('RUN')
        ret, frame = cap.read()
        if ret == True:
            bb = yolo(frame).xyxyn[0].cpu().tolist()
            if len(bb) > 0:
                for box in bb:
                    pred = box[-len(labels):]
                    label = labels[pred.index(max(pred))]
                    print(label)
                    width, height = frame.shape[1], frame.shape[0]
                    cords = get_bboxes(box, width, height)
                    cropped = frame[cords[1]:cords[3], cords[0]:cords[2]]
                    cv2.imshow('Cropped', cropped)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(args)
