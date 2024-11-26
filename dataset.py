import os
import torch
import torchvision.transforms as T
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from utils import intersection_over_union, plot_image

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
    """
    Load image and annotation information based on train.txt.
    """
    img_infos = []

    # Đường dẫn tới train.txt trong thư mục Main
    train_file = os.path.join(imgsets_dir, 'train.txt')  # Tệp chính để lấy img_id

    # Kiểm tra tệp train.txt có tồn tại hay không
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"File {train_file} does not exist.")

    # Đọc danh sách các img_id từ train.txt
    img_names = []
    with open(train_file, 'r') as f:
        for line in f:
            img_names.append(line.strip())
        # img_names = [line.strip() for line in f] # tham khảo

    # Duyệt qua từng img_id để lấy thông tin ảnh và annotation
    for img_name in img_names:
        img_info = {}

        # Đường dẫn tới file annotation (.xml) và ảnh (.jpg)
        anno_file = os.path.join(anno_dir, f'{img_name}.xml')
        img_file = os.path.join(img_dir, f'{img_name}.jpg')

        # Kiểm tra sự tồn tại của file annotation và file ảnh
        if not os.path.exists(anno_file):
            raise FileNotFoundError(f"Annotation file {anno_file} not found.")
        if not os.path.exists(img_file):
            raise FileNotFoundError(f"Image file {img_file} not found.")

        # Đọc thông tin annotation
        anno_info = ET.parse(anno_file)
        root = anno_info.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        img_info['width'] = width
        img_info['height'] = height
        img_info['img_id'] = img_name
        img_info['filename'] = img_file

        # Lấy thông tin bounding boxes và labels từ annotation
        detections = []
        for obj in root.findall('object'):
            det = {}
            label = label2idx[obj.find('name').text]
            bbox_info = obj.find('bndbox')
            bbox = [
                int(bbox_info.find('xmin').text) - 1,
                int(bbox_info.find('ymin').text) - 1,
                int(bbox_info.find('xmax').text) - 1,
                int(bbox_info.find('ymax').text) - 1,
            ]
            det['label'] = label
            det['bbox'] = bbox
            detections.append(det)

        img_info['detections'] = detections
        img_infos.append(img_info)

    print(f"Loaded {len(img_infos)} images and annotations from {train_file}.")
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
        # self.transforms = transforms

        # Chuyển đổi ảnh (resize, tensor hóa, normalize)
        self.transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        classes = get_classes(self.imgsets_dir)
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        self.images_info = load_img_and_anno(self.annotation_dir, self.img_dir, self.imgsets_dir, self.label2idx)

    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, index):
        img_info = self.images_info[index]
        img = Image.open(img_info['filename']).convert('RGB')
        
        # Áp dụng transform cho ảnh
        img_tensor = self.transforms(img)

        # Get annotations for this image
        bboxes = [detection['bbox'] for detection in img_info['detections']]
        labels = [detection['label'] for detection in img_info['detections']]
        
        # Convert image to tensor and normalize
        bboxes_tensor = torch.as_tensor(bboxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)        

        # Build Target for Yolo
        target_dim = 5 * self.B + self.C
        yolo_targets = torch.zeros(self.S, self.S, target_dim)

        # Xử lý lỗi: Clamp bounding boxes trong giới hạn ảnh
        h, w = img_info['height'], img_info['width']
        if len(bboxes) > 0:
            bboxes_tensor[:, [0, 2]] = torch.clamp(bboxes_tensor[:, [0, 2]], 0, w)
            bboxes_tensor[:, [1, 3]] = torch.clamp(bboxes_tensor[:, [1, 3]], 0, h)
        
        if len(bboxes) > 0:            
            # Height and width of grid cells is H // S
            cell_pixels = h // self.S
            
            # Convert x1y1x2y2 to xywh format
            box_widths = bboxes_tensor[:, 2] - bboxes_tensor[:, 0]
            box_heights = bboxes_tensor[:, 3] - bboxes_tensor[:, 1]
            box_center_x = bboxes_tensor[:, 0] + 0.5 * box_widths
            box_center_y = bboxes_tensor[:, 1] + 0.5 * box_heights

            # Get cell i,j from xc, yc
            """ # BAN
            box_i = torch.floor(box_center_x / cell_pixels).long()
            box_j = torch.floor(box_center_y / cell_pixels).long()
            """
            
            box_i = torch.clamp(torch.floor(box_center_x / cell_pixels).long(), min=0, max=self.S - 1)
            box_j = torch.clamp(torch.floor(box_center_y / cell_pixels).long(), min=0, max=self.S - 1)

            # xc offset from cell topleft
            box_xc_cell_offset = (box_center_x - box_i*cell_pixels) / cell_pixels
            box_yc_cell_offset = (box_center_y - box_j*cell_pixels) / cell_pixels

            # w, h targets normalized to 0-1
            box_w_label = box_widths / w
            box_h_label = box_heights / h

            
            # BAn

            # Update the target array for all bboxes and check limits
            for idx, b in enumerate(range(bboxes_tensor.size(0))):
                # Make target of the exact same shape as prediction
                for k in range(self.B):
                    s = 5 * k
                    
                    # target_ij = [xc_offset,yc_offset,sqrt(w),sqrt(h), conf, cls_label]
                    yolo_targets[box_j[idx], box_i[idx], s] = box_xc_cell_offset[idx]
                    yolo_targets[box_j[idx], box_i[idx], s+1] = box_yc_cell_offset[idx]
                    
                    #* 2 dòng dưới đây để so theo công thức 
                    # yolo_targets[box_j[idx], box_i[idx], s+2] = torch.sqrt(box_w_label[idx])
                    # yolo_targets[box_j[idx], box_i[idx], s+3] = torch.sqrt(box_h_label[idx])
                    
                    yolo_targets[box_j[idx], box_i[idx], s+2] = box_w_label[idx]
                    yolo_targets[box_j[idx], box_i[idx], s+3] = box_h_label[idx]
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
            bboxes_tensor /= torch.tensor([w, h, w, h], dtype=torch.float32).expand_as(bboxes_tensor)
            
        targets = {
            'bboxes': bboxes_tensor,
            'labels': labels_tensor,
            'yolo_targets': yolo_targets,
        }
        return img_tensor, targets, img_info['filename']         
    
# Custom collate function
def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    bboxes = [item[1]['bboxes'] for item in batch]
    labels = [item[1]['labels'] for item in batch]
    yolo_targets = [item[1]['yolo_targets'] for item in batch]

    # Stack images to create a batch
    images = torch.stack(images, dim=0)

    # Padding bounding boxes and labels if necessary
    bboxes_padded = torch.nn.utils.rnn.pad_sequence(bboxes, batch_first=True, padding_value=-1)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)

    # Stack yolo_targets since they should have a fixed shape
    yolo_targets = torch.stack(yolo_targets, dim=0)

    # Create a dictionary for targets
    targets = {
        'bboxes': bboxes_padded,
        'labels': labels_padded,
        'yolo_targets': yolo_targets,
    }

    return images, targets

