import torch
import pickle
import cv2
import numpy as np

args = {'labels_path': 'data/keys.json',
        'model_path': 'data/model_logit_cls'}
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def center_crop(img, dim):
    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
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
    return np.ascontiguousarray(im)

def main(args: dict):
    cap = cv2.VideoCapture(0)

    model = torch.load(args['model_path'])
    model.to('cuda').eval()
    with open(args['labels_path'], "rb") as fp:   # Unpickling
        labels = pickle.load(fp)

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            inputs = preprocess(frame)
            tensor_input = torch.Tensor(inputs)  # resizing image for torch model
            with torch.no_grad():
                logits = model(tensor_input[None, ...].to('cuda')).to('cpu')
            print(labels[torch.argmax(logits, dim=1)].item())
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(args)
