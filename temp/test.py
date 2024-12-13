import os
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from train import DEVICE
from utils import load_checkpoint
from train import transform, ROOT_DIR, BEST_CHECKPOINT_PATH, S, B, C
from model import Yolov1
from dataset import PascalVOC2012Dataset, get_class_dictionary

def test():
    check_point_path = BEST_CHECKPOINT_PATH

    if not os.path.exists(check_point_path):
        raise FileNotFoundError(f"Checkpoint not found at [{check_point_path}]")

    # Load mô hình
    model   = Yolov1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)
    model.eval()

    # Sử dụng hàm load_checkpoint để load trọng số
    model, _, _, _ = load_checkpoint(check_point_path, model, device=DEVICE, load_weights_only=True)
    print("Model weights loaded successfully for testing.")

    # load dataset to visualize and predict
    dataset = PascalVOC2012Dataset(root_dir=ROOT_DIR, split='trainval', transform=transform)

    # Truy xuất ngược lại để in tên trên bounding box3
    name2idx = get_class_dictionary(os.path.join(ROOT_DIR, 'ImageSets', 'Main'))
    idx2name = dict(zip([key for key in name2idx.values()], [value for value in name2idx.keys()]))

    for idx in range(dataset.__len__()):
        original_img, img, target = dataset.__getitem__(idx)
        img = img.unsqueeze(0).to('cuda')
        output = model(img)
        bboxes = []
        labels = []
        confs  = []
        factor = 1. / S
        W, H = original_img.size

        for i in range(S):
            for j in range(S):
                for k in range(B):
                    out  = output[0, i, j, k*5:k*5+5]
                    box  = out[:4]
                    conf = out[4]
                    cls  = output[0, i, j, B*5:]
                    a, b = torch.max(cls, dim=-1)
                    conf = a * conf
                    label = b
                    if (conf < 0.3):
                        continue
                    cx_cell = box[0]
                    cy_cell = box[1]
                    w       = box[2]
                    h       = box[3]

                    cx = (cx_cell + j) * factor
                    cy = (cy_cell + i) * factor

                    x1 = (cx - w/2)
                    y1 = (cy - h/2)
                    x2 = (cx + w/2)
                    y2 = (cy + h/2)

                    bboxes.append([x1, y1, x2, y2])
                    confs.append(conf)
                    labels.append(label)

        if len(bboxes) > 0:
            print(f"Number of bounding boxes before NMS: {len(bboxes)}")
            bboxes = torch.tensor(bboxes)
            confs  = torch.tensor(confs)
            labels = torch.tensor(labels)
            mask = torchvision.ops.nms(bboxes, confs, 0.5)

            bboxes = bboxes[mask]
            labels = labels[mask]

            print(f"Number of bounding boxes after NMS: {len(bboxes)}")
            draw = ImageDraw.Draw(original_img)
            font = ImageFont.load_default()

            for (box, label) in zip(bboxes, labels):
                x1 = int(box[0] * W)
                y1 = int(box[1] * H)
                x2 = int(box[2] * W)
                y2 = int(box[3] * H)

                print("box[0] {} and x1 {}, ".format(box[0], x1))
                print("box[1] {} and y1 {}, ".format(box[1], y1))
                print("box[2] {} and x2 {}, ".format(box[2], x2))
                print("box[3] {} and y2 {}, ".format(box[3], y2))

                draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=2)
                draw.text((x1, y1), idx2name[int(label.item())], font=font, fill='black')

            plt.imshow(original_img)
            plt.show()

    print("\nTesting completed!")

if __name__ == '__main__':
    test()