import os
import torch
import torchvision.transforms as T
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset

def get_classes(image_sets_dir):
    classes = []

    # Lấy tất cả các file .txt trong thư mục
    class_files = [f for f in os.listdir(image_sets_dir) if f.endswith('_train.txt')] # ['aeroplane_train.txt', 'bicycle_train.txt', 'bird_train.txt', 'boat_train.txt', 'bottle_train.txt', 'bus_train.txt', 'car_train.txt', 'cat_train.txt', 'chair_train.txt', 'cow_train.txt', 'diningtable_train.txt', 'dog_train.txt', 'horse_train.txt', 'motorbike_train.txt', 'person_train.txt', 'pottedplant_train.txt', 'sheep_train.txt', 'sofa_train.txt', 'train_train.txt', 'tvmonitor_train.txt']
    
    # Trích xuất tên class từ tên file (bỏ đuôi _train.txt)
    for class_file in sorted(class_files):
        class_name = class_file.replace('_train.txt', '')
        classes.append(class_name)
    
    # Loại bỏ các file trùng lặp và sắp xếp
    classes = sorted(list(set(classes)))    
    return classes


def load_img_and_anno(anno_dir, img_dir, imgsets_dir, label2idx):
    img_infos = []

    for img_set in imgsets_dir:
        img_names = []
        # Fetch all image names in txt file for this imageset
        for line in open(os.path.join(img_set, '{}.txt'.format('train'))):
            img_names.append(line.strip())

        for img_name in img_names:
            img_info = {}

            # Đọc anno
            anno_file = os.path.join(anno_dir, f'{img_name}.xml')
            anno_info = ET.parse(anno_file)  # đọc file .xml
            root = anno_info.getroot()   #  trả về phần gốc của cây XML, cho phép truy cập vào các thẻ (tag) con bên trong.
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_info['width'] = width
            img_info['height'] = height
            img_info['img_id'] = os.path.basename(anno_file).split('.xml')[0]
            img_info['filename'] = os.path.join(img_dir, '{}.jpg'.format(img_info['img_id']))

            detections = []

            for obj in anno_info.findall('object'):
                det = {}
                label = label2idx[obj.find('name').text]
                bbox_info = obj.find('bndbox')
                bbox = [
                    int(float(bbox_info.find('xmin').text))-1,
                    int(float(bbox_info.find('ymin').text))-1,
                    int(float(bbox_info.find('xmax').text))-1,
                    int(float(bbox_info.find('ymax').text))-1
                ]

                det['label'] = label
                det['bbox'] = bbox
                detections.append(det)

            img_info['detections'] = detections
            img_infos.append(img_info)

    return img_infos


class PascalVOC2012Dataset(Dataset):
    def __init__(self, root_dir, img_size = 448, S=7, B=2, C=20):
        self.root_dir = root_dir
        self.annotation_dir = os.path.join(root_dir, "Annotations")
        self.img_dir = os.path.join(root_dir, "JPEGImages")
        self.imgsets_dir = os.path.join(root_dir, "ImageSets", "Main")
        self.S = S
        self.B = B
        self.C = C


        self.transforms = T.Compose([T.Resize((img_size, img_size)), T.ToTensor])


        classes = get_classes(self.imgsets_dir)
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        self.images_info = load_img_and_anno(self.annotation_dir, self.img_dir, self.imgsets_dir, self.label2idx)

    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, index):
        img_info = self.images_info[index]
        img = Image.open(img_info['filename']).convert('RGB')

        # Get annotations for this image
        bboxes = [detection['bbox'] for detection in img_info['detections']]
        labels = [detection['label'] for detection in img_info['detections']]

        # Convert image to tensor and normalize
        img_tensor = torch.from_numpy(img / 255.).permute((2, 0, 1)).float()
        img_tensor_channel_0 = (torch.unsqueeze(img_tensor[0], 0) - 0.485) / 0.229
        img_tensor_channel_1 = (torch.unsqueeze(img_tensor[1], 0) - 0.456) / 0.224
        img_tensor_channel_2 = (torch.unsqueeze(img_tensor[2], 0) - 0.406) / 0.225
        img_tensor = torch.cat((img_tensor_channel_0,
                               img_tensor_channel_1,
                               img_tensor_channel_2), 0)
        bboxes_tensor = torch.as_tensor(bboxes)
        labels_tensor = torch.as_tensor(labels)

        # Build Target for Yolo
        target_dim = 5 * self.B + self.C
        h, w = img.shape[:2]
        yolo_targets = torch.zeros(self.S, self.S, target_dim)

        # Height and width of grid cells is H // S
        cell_pixels = h // self.S

        if len(bboxes) > 0:
            # Convert x1y1x2y2 to xywh format
            box_widths = bboxes_tensor[:, 2] - bboxes_tensor[:, 0]
            box_heights = bboxes_tensor[:, 3] - bboxes_tensor[:, 1]
            box_center_x = bboxes_tensor[:, 0] + 0.5 * box_widths
            box_center_y = bboxes_tensor[:, 1] + 0.5 * box_heights

            # Get cell i,j from xc, yc
            box_i = torch.floor(box_center_x / cell_pixels).long()
            box_j = torch.floor(box_center_y / cell_pixels).long()

            # xc offset from cell topleft
            box_xc_cell_offset = (box_center_x - box_i*cell_pixels) / cell_pixels
            box_yc_cell_offset = (box_center_y - box_j*cell_pixels) / cell_pixels

            # w, h targets normalized to 0-1
            box_w_label = box_widths / w
            box_h_label = box_heights / h

            # Update the target array for all bboxes
            for idx, b in enumerate(range(bboxes_tensor.size(0))):
                # Make target of the exact same shape as prediction
                for k in range(self.B):
                    s = 5 * k
                    # target_ij = [xc_offset,yc_offset,sqrt(w),sqrt(h), conf, cls_label]
                    yolo_targets[box_j[idx], box_i[idx], s] = box_xc_cell_offset[idx]
                    yolo_targets[box_j[idx], box_i[idx], s+1] = box_yc_cell_offset[idx]
                    yolo_targets[box_j[idx], box_i[idx], s+2] = box_w_label[idx].sqrt()
                    yolo_targets[box_j[idx], box_i[idx], s+3] = box_h_label[idx].sqrt()
                    yolo_targets[box_j[idx], box_i[idx], s+4] = 1.0
                label = int(labels[b])
                cls_target = torch.zeros((self.C,))
                cls_target[label] = 1.
                yolo_targets[box_j[idx], box_i[idx], 5 * self.B:] = cls_target
        # For training, we use yolo_targets(xoffset, yoffset, sqrt(w), sqrt(h))
        # For evaluation we use bboxes_tensor (x1, y1, x2, y2)
        # Below we normalize bboxes tensor to be between 0-1
        # as thats what evaluation script expects so (x1/w, y1/h, x2/w, y2/h)
        if len(bboxes) > 0:
            bboxes_tensor /= torch.Tensor([[w, h, w, h]]).expand_as(bboxes_tensor)
        targets = {
            'bboxes': bboxes_tensor,
            'labels': labels_tensor,
            'yolo_targets': yolo_targets,
        }
        return img_tensor, targets, img_info['filename']         