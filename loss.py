import torch
import torch.nn as nn

# box = [x, y, w, h]
# bbox = [x1, y1, w1, h1, confident_score1, x2, y2, w2, h2, confident_score2, 20 classes]

def get_iou(det, gt): # (detected boundingbox, ground truth boundingbox)
    r"""
    IOU between two sets of boxes
    """
    # Area of boxes (x2-x1)*(y2-y1)
    det_area = (det[..., 2] - det[..., 0]) * (det[..., 3] - det[..., 1])
    gt_area = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])

    # Get top left x1,y1 coordinate
    x_left = torch.max(det[..., 0], gt[..., 0])
    y_top = torch.max(det[..., 1], gt[..., 1])
    # Get bottom right x2,y2 coordinate
    x_right = torch.min(det[..., 2], gt[..., 2])
    y_bottom = torch.min(det[..., 3], gt[..., 3])

    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union = det_area.clamp(min=0) + gt_area.clamp(min=0) - intersection_area
    iou = intersection_area / (union + 1E-6)
    return iou


class YOLOv1Loss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(YOLOv1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
    
    def forward(self, preds, targets, use_sigmoid=False):
        batch_size = preds.size(0)

        # preds -> (Batch, S, S, 5B+C)
        preds = preds.reshape(batch_size, self.S, self.S, 5*self.B + self.C)

        # Generally sigmoid leads to quicker convergence
        if use_sigmoid:
            preds[..., :5 * self.B] = torch.nn.functional.sigmoid(preds[..., :5 * self.B])

        # Shifts for all grid cell locations.
        # Will use these for converting x_center_offset/y_center_offset
        # values to x1/y1/x2/y2(normalized 0-1)
        # S cells = 1 => each cell adds 1/S pixels of shift

        # Tọa độ dịch chuyển cho từng hàng của lưới
        shifts_x = torch.arange(0, self.S,
                                dtype=torch.int32,
                                device=preds.device) * 1 / float(self.S)    # tensor([0.0000, 0.1429, 0.2857, 0.4286, 0.5714, 0.7143, 0.8571])
        # Tọa độ dịch chuyển cho từng cột của lưới
        shifts_y = torch.arange(0, self.S,
                                dtype=torch.int32,
                                device=preds.device) * 1 / float(self.S)    # tensor([0.0000, 0.1429, 0.2857, 0.4286, 0.5714, 0.7143, 0.8571])
        
        # Create a grid using these shifts
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        # shifts -> (1, S, S, B)
        # repeat: nhân bản để có 2 lưới tọa độ dịch chuyển cho 2 bbox
        shifts_x = shifts_x.reshape((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B)
        shifts_y = shifts_y.reshape((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B)

        # pred_boxes -> (Batch_size, S, S, B, 5)
        # 5: x, y, w, h, confidence_score
        pred_boxes = preds[..., :5 * self.B].reshape(batch_size, self.S, self.S, self.B, -1)

        # xc_offset yc_offset w h -> x1 y1 x2 y2 (normalized 0-1)
        # x_center = (xc_offset / S + shift_x)
        # x1 = x_center - 0.5 * w
        # x2 = x_center + 0.5 * w
        pred_boxes_x1 = ((pred_boxes[..., 0]/self.S + shifts_x)
                         - 0.5*(pred_boxes[..., 2]))
        pred_boxes_x1 = pred_boxes_x1[..., None]
        pred_boxes_y1 = ((pred_boxes[..., 1]/self.S + shifts_y)
                         - 0.5*(pred_boxes[..., 3]))
        pred_boxes_y1 = pred_boxes_y1[..., None]

        pred_boxes_x2 = ((pred_boxes[..., 0]/self.S + shifts_x)
                         + 0.5*(pred_boxes[..., 2]))
        pred_boxes_x2 = pred_boxes_x2[..., None]
        pred_boxes_y2 = ((pred_boxes[..., 1]/self.S + shifts_y)
                         + 0.5*(pred_boxes[..., 3]))
        pred_boxes_y2 = pred_boxes_y2[..., None]

        pred_boxes_x1y1x2y2 = torch.cat([
            pred_boxes_x1,
            pred_boxes_y1,
            pred_boxes_x2,
            pred_boxes_y2], dim=-1)
        

        # target_boxes -> (Batch_size, S, S, B, 5)
        target_boxes = targets[..., :5*self.B].reshape(batch_size, self.S, self.S, self.B, -1)
        target_boxes_x1 = ((target_boxes[..., 0] / self.S + shifts_x)
                           - 0.5 * (target_boxes[..., 2]))
        target_boxes_x1 = target_boxes_x1[..., None]
        target_boxes_y1 = ((target_boxes[..., 1] / self.S + shifts_y)
                           - 0.5 * (target_boxes[..., 3]))
        target_boxes_y1 = target_boxes_y1[..., None]

        target_boxes_x2 = ((target_boxes[..., 0] / self.S + shifts_x)
                           + 0.5 * (target_boxes[..., 2]))
        target_boxes_x2 = target_boxes_x2[..., None]
        target_boxes_y2 = ((target_boxes[..., 1] / self.S + shifts_y)
                           + 0.5 * (target_boxes[..., 3]))
        target_boxes_y2 = target_boxes_y2[..., None]

        target_boxes_x1y1x2y2 = torch.cat([
            target_boxes_x1,
            target_boxes_y1,
            target_boxes_x2,
            target_boxes_y2
        ], dim=-1)

        # iou -> (Batch_size, S, S, B)
        iou = get_iou(pred_boxes_x1y1x2y2, target_boxes_x1y1x2y2)

        # max_iou_val/max_iou_idx -> (Batch_size, S, S, 1)
        max_iou_val, max_iou_idx = iou.max(dim=-1, keepdim=True)

        #########################
        # Indicator Definitions #
        #########################
        # before max_iou_idx -> (Batch_size, S, S, 1) Eg [[0], [1], [0], [0]]
        # after repeating max_iou_idx -> (Batch_size, S, S, B)
        # Eg. [[0, 0], [1, 1], [0, 0], [0, 0]] assuming B = 2
        max_iou_idx = max_iou_idx.repeat(1, 1, 1, self.B)
        # bb_idxs -> (Batch_size, S, S, B)
        #  Eg. [[0, 1], [0, 1], [0, 1], [0, 1]] assuming B = 2
        bb_idxs = (torch.arange(self.B).reshape(1, 1, 1, self.B).expand_as(max_iou_idx)
                   .to(preds.device))
        # is_max_iou_box -> (Batch_size, S, S, B)
        # Eg. [[True, False], [False, True], [True, False], [True, False]]
        # only the index which is max iou boxes index will be 1 rest all 0
        is_max_iou_box = (max_iou_idx == bb_idxs).long()

        # obj_indicator -> (Batch_size, S, S, 1)
        obj_indicator = targets[..., 4:5]


        #######################
        # Classification Loss #
        #######################
        cls_target = targets[..., 5 * self.B:]
        cls_preds = preds[..., 5 * self.B:]
        cls_mse = (cls_preds - cls_target) ** 2
        # Only keep losses from cells with object assigned
        cls_mse = (obj_indicator * cls_mse).sum()


        ###################################################### 
        # Objectness Loss (For responsible predictor boxes ) #
        ######################################################
        # indicator is now object_cells * is_best_box
        is_max_box_obj_indicator = is_max_iou_box * obj_indicator
        obj_mse = (pred_boxes[..., 4] - max_iou_val) ** 2
        # Only keep losses from boxes of cells with object assigned
        # and that box which is the responsible predictor
        obj_mse = (is_max_box_obj_indicator * obj_mse).sum()

        #####################
        # Localization Loss #
        #####################
        x_mse = (pred_boxes[..., 0] - target_boxes[..., 0]) ** 2
        # Only keep losses from boxes of cells with object assigned
        # and that box which is the responsible predictor
        x_mse = (is_max_box_obj_indicator * x_mse).sum()

        y_mse = (pred_boxes[..., 1] - target_boxes[..., 1]) ** 2
        y_mse = (is_max_box_obj_indicator * y_mse).sum()
        w_sqrt_mse = (torch.sqrt(pred_boxes[..., 2]) - torch.sqrt(target_boxes[..., 2])) ** 2
        w_sqrt_mse = (is_max_box_obj_indicator * w_sqrt_mse).sum()
        h_sqrt_mse = (torch.sqrt(pred_boxes[..., 3]) - torch.sqrt(target_boxes[..., 3])) ** 2
        h_sqrt_mse = (is_max_box_obj_indicator * h_sqrt_mse).sum()


        ################################################# 
        # Objectness Loss
        # For boxes of cells assigned with object that
        # aren't responsible predictor boxes
        # and for boxes of cell not assigned with object
        #################################################
        no_object_indicator = 1 - is_max_box_obj_indicator
        no_obj_mse = (pred_boxes[..., 4] - torch.zeros_like(pred_boxes[..., 4])) ** 2
        no_obj_mse = (no_object_indicator * no_obj_mse).sum()


        ##############
        # Total Loss #
        ##############
        loss = self.lambda_coord*(x_mse + y_mse + w_sqrt_mse + h_sqrt_mse)
        loss += cls_mse + obj_mse
        loss += self.lambda_noobj*no_obj_mse
        loss = loss / batch_size
        return loss
    