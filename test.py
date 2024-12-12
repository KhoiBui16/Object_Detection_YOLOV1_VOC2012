import os
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from model import Yolov1
from train import ROOT_DIR
from train import CHECKPOINT_PATH
from train import B, S, C
from dataset import PascalVOC2012Dataset, get_class_dictionary

def test():
    dataset = PascalVOC2012Dataset(root_dir=ROOT_DIR, split='train')
    model   = Yolov1(split_size=S, num_boxes=B, num_classes=C).to('cuda')
    model.eval()
    check_point = CHECKPOINT_PATH
    model.load_state_dict(torch.load(check_point, weights_only=True))
    name2idx = get_class_dictionary(os.path.join(ROOT_DIR, 'ImageSets', 'Main'))
    
    # Truy xuất ngược lại để in tên trên bounding box
    idx2name = dict(zip([key for key in name2idx.values()], [value for value in name2idx.keys()]))
    
    for idx in range(dataset.__len__()):
        original_img, img, target = dataset.__getitem__(idx)
        img    = img.unsqueeze(0).to('cuda')
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
                    # Giảm ngưỡng conf để vẽ được nhiều box hơn
                    if conf < 0.2:  # Điều chỉnh ngưỡng để giữ lại nhiều box hơn
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
            print(f"Number of bounding boxes before NMS: {len(bboxes)}")  # In số lượng box ban đầu
            bboxes = torch.tensor(bboxes)
            confs  = torch.tensor(confs)
            labels = torch.tensor(labels)
            mask = torchvision.ops.nms(bboxes, confs, 0.5)

            bboxes = bboxes[mask]
            labels = labels[mask]

            print(f"Number of bounding boxes after NMS: {len(bboxes)}")  # In số lượng box sau NMS
            draw = ImageDraw.Draw(original_img)
            font = ImageFont.truetype("arial.ttf", size=15)  # Sử dụng font TTF thay cho mặc định

            for (box, label) in zip(bboxes, labels):
                x1 = int(box[0] * W)
                y1 = int(box[1] * H)
                x2 = int(box[2] * W)
                y2 = int(box[3] * H)

                draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=2)
                draw.text((x1, y1), idx2name[label.item()], font=font)

            plt.imshow(original_img)
            plt.show()

if __name__ == '__main__':
    test()
