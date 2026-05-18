import  numpy as np
import torch.nn as nn
import torch
from skimage import measure
import  numpy
import json
import os
import tempfile
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])



class PD_FA():
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.target= np.zeros(self.bins + 1)
    def update(self, preds, labels):

        for iBin in range(self.bins+1):
            score_thresh = iBin * (255/self.bins)
            predits  = np.array((preds > score_thresh).cpu()).astype('int64')
            predits  = np.reshape (predits,  (256,256))
            labelss = np.array((labels).cpu()).astype('int64') # P
            labelss = np.reshape (labelss , (256,256))

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin]    += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin]+=np.sum(self.dismatch)
            self.PD[iBin]+=len(self.distance_match)

    def get(self,img_num):

        Final_FA =  self.FA / ((256 * 256) * img_num)
        Final_PD =  self.PD /self.target

        return Final_FA,Final_PD


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])

class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def get(self):

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0




def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union

def bbox_iou(boxA, boxB):
    # box format: [min_row, min_col, max_row, max_col]
    rA_min, cA_min, rA_max, cA_max = boxA
    rB_min, cB_min, rB_max, cB_max = boxB

    inter_rmin = max(rA_min, rB_min)
    inter_cmin = max(cA_min, cB_min)
    inter_rmax = min(rA_max, rB_max)
    inter_cmax = min(cA_max, cB_max)

    if inter_rmax > inter_rmin and inter_cmax > inter_cmin:
        inter_area = (inter_rmax - inter_rmin) * (inter_cmax - inter_cmin)
    else:
        inter_area = 0

    areaA = (rA_max - rA_min) * (cA_max - cA_min)
    areaB = (rB_max - rB_min) * (cB_max - cB_min)
    union_area = areaA + areaB - inter_area
    if union_area == 0:
        return 0
    return inter_area / float(union_area)

class BoundingBoxMetric():
    def __init__(self, bins, iou_thresh=0.1):
        super(BoundingBoxMetric, self).__init__()
        self.bins = bins
        self.iou_thresh = iou_thresh
        self.tp_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.fn_arr = np.zeros(self.bins+1)

    def update(self, preds, labels):
        # preds and labels have shape (B, 1, H, W)
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            predits = (torch.sigmoid(preds) > score_thresh).cpu().numpy().astype('int64')
            labelss = (labels > 0).cpu().numpy().astype('int64')
            
            batch_size = predits.shape[0]
            for b in range(batch_size):
                # Handle possible varying tensor shapes
                if len(predits.shape) == 4:
                    pred_img = predits[b, 0, :, :]
                    label_img = labelss[b, 0, :, :]
                elif len(predits.shape) == 3:
                    pred_img = predits[b, :, :]
                    label_img = labelss[b, :, :]
                else:
                    pred_img = predits
                    label_img = labelss

                image = measure.label(pred_img, connectivity=2)
                coord_image = measure.regionprops(image)
                
                label_l = measure.label(label_img, connectivity=2)
                coord_label = measure.regionprops(label_l)
                
                matched_preds = set()
                matched_labels = set()
                
                for i_gt, props_gt in enumerate(coord_label):
                    bbox_gt = props_gt.bbox
                    for i_pred, props_pred in enumerate(coord_image):
                        if i_pred in matched_preds:
                            continue
                        bbox_pred = props_pred.bbox
                        iou = bbox_iou(bbox_pred, bbox_gt)
                        if iou >= self.iou_thresh:
                            matched_preds.add(i_pred)
                            matched_labels.add(i_gt)
                            break  # Move to next GT
                            
                self.tp_arr[iBin] += len(matched_preds)
                self.fp_arr[iBin] += len(coord_image) - len(matched_preds)
                self.fn_arr[iBin] += len(coord_label) - len(matched_labels)

    def get(self):
        precision = self.tp_arr / (self.tp_arr + self.fp_arr + 1e-6)
        recall = self.tp_arr / (self.tp_arr + self.fn_arr + 1e-6)
        return recall, precision

    def reset(self):
        self.tp_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.fn_arr = np.zeros(self.bins+1)

class COCOEvaluator():
    def __init__(self, iou_thresh=0.1, score_thresh=0.5):
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        
        self.annotations = []
        self.images = []
        self.predictions = []
        
        self.ann_id = 1
        self.cat_id = 1
        self.categories = [{"id": self.cat_id, "name": "target", "supercategory": "none"}]
        self.image_ids_seen = set()

    def update(self, preds, labels, img_ids):
        preds_sigmoid = torch.sigmoid(preds).cpu().numpy()
        preds_bin = (preds_sigmoid > self.score_thresh).astype(np.uint8)
        labels_bin = (labels > 0).cpu().numpy().astype(np.uint8)
        
        batch_size = preds_sigmoid.shape[0]
        
        for b in range(batch_size):
            img_id_str = str(img_ids[b])
            try:
                numeric_part = "".join(filter(str.isdigit, img_id_str))
                if numeric_part:
                    img_id_int = int(numeric_part)
                else:
                    img_id_int = abs(hash(img_id_str)) % (10 ** 8)
            except:
                img_id_int = abs(hash(img_id_str)) % (10 ** 8)
            
            while img_id_int in self.image_ids_seen:
                 img_id_int += 1
            self.image_ids_seen.add(img_id_int)
                
            self.images.append({
                "id": img_id_int,
                "file_name": img_id_str + ".png",
                "width": 256,
                "height": 256
            })
            
            gt_img = labels_bin[b, 0, :, :] if len(labels_bin.shape) == 4 else (labels_bin[b, :, :] if len(labels_bin.shape) == 3 else labels_bin)
            gt_labeled = measure.label(gt_img, connectivity=2)
            for region in measure.regionprops(gt_labeled):
                min_row, min_col, max_row, max_col = region.bbox
                width = max_col - min_col
                height = max_row - min_row
                self.annotations.append({
                    "id": self.ann_id,
                    "image_id": img_id_int,
                    "category_id": self.cat_id,
                    "bbox": [min_col, min_row, width, height],
                    "area": width * height,
                    "iscrowd": 0
                })
                self.ann_id += 1
                
            pred_bin_img = preds_bin[b, 0, :, :] if len(preds_bin.shape) == 4 else (preds_bin[b, :, :] if len(preds_bin.shape) == 3 else preds_bin)
            pred_prob_img = preds_sigmoid[b, 0, :, :] if len(preds_sigmoid.shape) == 4 else (preds_sigmoid[b, :, :] if len(preds_sigmoid.shape) == 3 else preds_sigmoid)
            
            pred_labeled = measure.label(pred_bin_img, connectivity=2)
            for region in measure.regionprops(pred_labeled):
                min_row, min_col, max_row, max_col = region.bbox
                width = max_col - min_col
                height = max_row - min_row
                mask = (pred_labeled == region.label)
                score = float(pred_prob_img[mask].max())
                
                self.predictions.append({
                    "image_id": img_id_int,
                    "category_id": self.cat_id,
                    "bbox": [min_col, min_row, width, height],
                    "score": score
                })
                
    def get(self):
        gt_json = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }
        if len(self.predictions) == 0:
            return 0.0, 0.0, 0.0
            
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as gt_file:
            json.dump(gt_json, gt_file)
            gt_path = gt_file.name
            
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as pred_file:
            json.dump(self.predictions, pred_file)
            pred_path = pred_file.name
            
        map_score = 0.0
        best_precision = 0.0
        recall = 0.0
        
        try:
            coco_gt = COCO(gt_path)
            coco_dt = coco_gt.loadRes(pred_path)
            
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.params.iouThrs = np.array([self.iou_thresh])
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            map_score = float(coco_eval.stats[0])
            recall = float(coco_eval.stats[8])
            
            precisions = coco_eval.eval["precision"]
            valid_precisions = precisions[0, :, 0, 0, -1] 
            valid_precisions = valid_precisions[valid_precisions > -1]
            if len(valid_precisions) > 0:
                best_precision = float(valid_precisions[-1])
                
        except Exception as e:
            print(f"Error during COCO evaluation: {e}")
            
        finally:
            if os.path.exists(gt_path):
                try:
                    os.remove(gt_path)
                except:
                    pass
            if os.path.exists(pred_path):
                try:
                    os.remove(pred_path)
                except:
                    pass
                
        return map_score, best_precision, recall

    def save_final_json(self, save_dir, file_prefix):
        gt_json = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }
        gt_path = os.path.join(save_dir, f"{file_prefix}_coco_gt.json")
        pred_path = os.path.join(save_dir, f"{file_prefix}_coco_pred.json")
        with open(gt_path, 'w') as f:
            json.dump(gt_json, f)
        with open(pred_path, 'w') as f:
            json.dump(self.predictions, f)
        print(f"Saved COCO GT to {gt_path} and Preds to {pred_path}")
