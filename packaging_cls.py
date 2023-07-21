from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
import argparse
import torch
import cv2
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--labels_path', type=str, default='data/keys.pt')
parser.add_argument('--model_path', type=str, default='data/model_packaging')
parser.add_argument('--od_model_path', type=str, default='C:/Deep Learning/cart_product_detector/data/od_model__packaging/od_model__packaging')
parser.add_argument('--extractor_path', type=str, default='C:/Deep Learning/cart_product_detector/data/extractor_packaging/extractor_packaging')
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

def main(args: dict):
    cap = cv2.VideoCapture(0)

    od_model = AutoModelForObjectDetection.from_pretrained(args.od_model_path, local_files_only=True)
    od_model.to('cuda').eval()
    labels = torch.load(args.labels_path)
    model_cls = torch.load(args.model_path)
    model_cls.to('cuda').eval()
    extractor = AutoFeatureExtractor.from_pretrained(args.extractor_path)

    while (cap.isOpened()):
        print('RUN')
        ret, frame = cap.read()
        if ret == True:
            inputs = extractor(frame, return_tensors="pt")
            with torch.no_grad():
                pixel_values = inputs.pixel_values.to('cuda')
                mask = inputs.pixel_mask.to('cuda')
                outputs = od_model(pixel_values, mask)
            #print(frame.shape)
            target_sizes = torch.tensor([[480, 640]])
            results = extractor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
            if results['scores'].shape[0] != 0:
                for score, _, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = [round(i, 2) for i in box.tolist()]
                    frame_img = Image.fromarray(frame, 'RGB')
                    croppped = np.array(frame_img.crop(box))
                    cv2.imshow('Pack', croppped)
                    input_tensor = preprocess(croppped).to('cuda')
                    with torch.no_grad():
                        logits = model_cls(input_tensor[None, ...])
                    print(labels[torch.argmax(logits, dim=1).item()])
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(args)
