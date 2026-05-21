from PIL import Image, ImageOps, ImageFilter
import platform, os
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import  torch
from torch.nn import init
from datetime import datetime
import argparse
import shutil
from  matplotlib import pyplot as plt
import cv2
from skimage import measure as sk_measure
class TrainSetLoader(Dataset):


    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id ,base_size=512,crop_size=480,transform=None,suffix='.png'):
        super(TrainSetLoader, self).__init__()

        self.transform = transform
        self._items = img_id
        self.masks = dataset_dir+'/'+'masks'
        self.images = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        return img, mask

    def __getitem__(self, idx):

        img_id     = self._items[idx]                        # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images+'/'+img_id+self.suffix   # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix

        img = Image.open(img_path).convert('RGB')         ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)

        # synchronized transform
        img, mask = self._sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32')/ 255.0


        return img, torch.from_numpy(mask) #img_id[-1]

    def __len__(self):
        return len(self._items)


class TestSetLoader(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id,transform=None,base_size=512,crop_size=480,suffix='.png'):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask

    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images+'/'+img_id+self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix
        img  = Image.open(img_path).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)
        # synchronized transform
        img, mask = self._testval_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0


        return img, torch.from_numpy(mask)  # img_id[-1]

    def __len__(self):
        return len(self._items)

class DemoLoader (Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, transform=None,base_size=512,crop_size=480,suffix='.png'):
        super(DemoLoader, self).__init__()
        self.transform = transform
        self.images    = dataset_dir
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _demo_sync_transform(self, img):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)

        # final transform
        img = np.array(img)
        return img

    def img_preprocess(self):
        img_path   =  self.images
        img  = Image.open(img_path).convert('RGB')

        # synchronized transform
        img  = self._demo_sync_transform(img)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img



def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal(m.weight.data)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_ckpt(state, save_path, filename):
    torch.save(state, os.path.join(save_path,filename))

def save_train_log(args, save_dir):
    dict_args=vars(args)
    args_key=list(dict_args.keys())
    args_value = list(dict_args.values())
    with open('result/%s/train_log.txt'%save_dir ,'w') as  f:
        now = datetime.now()
        f.write("time:--")
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write('\n')
        for i in range(len(args_key)):
            f.write(args_key[i])
            f.write(':--')
            f.write(str(args_value[i]))
            f.write('\n')
    return

def save_model_and_result(dt_string, epoch,train_loss, test_loss, best_metric, recall, precision, save_mIoU_dir, save_other_metric_dir, bbox_recall=None, bbox_precision=None, bbox_mAP=None):

    with open(save_mIoU_dir, 'a') as f:
        f.write('{} - {:04d}:\t - train_loss: {:04f}:\t - test_loss: {:04f}:\t BBox_mAP {:.4f}\n' .format(dt_string, epoch,train_loss, test_loss, best_metric))
    with open(save_other_metric_dir, 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epoch))
        f.write('\n')
        f.write('===== BOUNDING BOX METRICS (PRIMARY - IoU=0.1) =====\n')
        f.write('BBox_Recall:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('BBox_Precision:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')
        
        if bbox_mAP is not None:
            f.write('BBox_mAP: ')
            f.write(str(round(bbox_mAP, 8)))
            f.write('\n')

def save_model(current_mAP, best_mAP, save_dir, save_prefix, train_loss, test_loss, recall, precision, epoch, net, mean_iou=None):
    if current_mAP > best_mAP:
        save_mAP_dir = 'result/' + save_dir + '/' + save_prefix + '_best_mAP.log'
        save_other_metric_dir = 'result/' + save_dir + '/' + save_prefix + '_best_metrics.log'
        value_result_dir = 'result/' + save_dir + '/value_result'
        
        from datetime import datetime
        import os
        
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        
        os.makedirs(value_result_dir, exist_ok=True)
        
        # 1. Metin log dosyalarına kaydet
        save_model_and_result(dt_string, epoch, train_loss, test_loss, current_mAP,
                              recall, precision, save_mAP_dir, save_other_metric_dir, bbox_mAP=current_mAP)
        
        # 2. COCO JSON Formatında Kaydet
        # pycocotools evaluation is handled internally and printed. Final JSON saving happens during test phase.
        
        # 3. Model ağırlıklarını .pth.tar olarak kaydet
        save_ckpt({
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'best_mAP': current_mAP,
            'mean_iou': mean_iou
        }, save_path='result/' + save_dir,
            filename=f'Best_mAP_{current_mAP:.4f}_{save_prefix}_epoch{epoch}.pth.tar')
        
        return current_mAP 
    
    return best_mAP

def save_result_for_test(dataset_dir, st_model, epochs, best_metric, recall, precision, bbox_recall=None, bbox_precision=None, bbox_mAP=None):
    with open(dataset_dir + '/' + 'value_result'+'/' + st_model +'_best_metric.log', 'w') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('{} - {:04d}:\t{:.4f}\n'.format(dt_string, epochs, best_metric))

    with open(dataset_dir + '/' + 'value_result'+'/'+ st_model + '_best_other_metric.log', 'w') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write('-')
        f.write(str(epochs))
        f.write('\n')
        f.write('===== BOUNDING BOX METRICS (PRIMARY - IoU=0.1) =====\n')
        f.write('BBox_Recall:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('BBox_Precision:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')
        
        if bbox_mAP is not None:
            f.write('BBox_mAP: ')
            f.write(str(round(bbox_mAP, 8)))
            f.write('\n')
    return

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def make_dir(deep_supervision, dataset, model):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    if deep_supervision:
        save_dir = "%s_%s_%s_wDS" % (dataset, model, dt_string)
    else:
        save_dir = "%s_%s_%s_woDS" % (dataset, model, dt_string)
    os.makedirs('result/%s' % save_dir, exist_ok=True)
    return save_dir

def total_visulization_generation(dataset_dir, mode, test_txt, suffix, target_image_path, target_dir):
    source_image_path = dataset_dir + '/images'

    txt_path = test_txt
    ids = []
    with open(txt_path, 'r') as f:
        ids += [line.strip() for line in f.readlines()]

    for i in range(len(ids)):
        source_image = source_image_path + '/' + ids[i] + suffix
        target_image = target_image_path + '/' + ids[i] + suffix
        shutil.copy(source_image, target_image)
    for i in range(len(ids)):
        source_image = target_image_path + '/' + ids[i] + suffix
        img = Image.open(source_image)
        img = img.resize((256, 256), Image.ANTIALIAS)
        img.save(source_image)
    for m in range(len(ids)):
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        img = plt.imread(target_image_path + '/' + ids[m] + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Raw Imamge", size=11)

        plt.subplot(1, 3, 2)
        img = plt.imread(target_image_path + '/' + ids[m] + '_GT' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Ground Truth", size=11)

        plt.subplot(1, 3, 3)
        img = plt.imread(target_image_path + '/' + ids[m] + '_Pred' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Predicts", size=11)

        plt.savefig(target_dir + '/' + ids[m].split('.')[0] + "_fuse" + suffix, facecolor='w', edgecolor='red')



def make_visulization_dir(target_image_path, target_dir):
    import time as _time
    for path in [target_image_path, target_dir]:
        if os.path.exists(path):
            for attempt in range(5):
                try:
                    shutil.rmtree(path)
                    break
                except PermissionError:
                    _time.sleep(1)
            else:
                shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

def save_Pred_GT(pred, labels, target_image_path, val_img_ids, num, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)


def save_Pred_GT_visulize(pred, img_demo_dir, img_demo_index, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) +suffix)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    img = plt.imread(img_demo_dir + '/' + img_demo_index + suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Raw Imamge", size=11)

    plt.subplot(1, 2, 2)
    img = plt.imread(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) +suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Predicts", size=11)


    plt.savefig(img_demo_dir + '/' + img_demo_index + "_fuse" + suffix, facecolor='w', edgecolor='red')
    plt.show()



def save_and_visulize_demo(pred, labels, target_image_path, val_img_ids, num, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)

    return


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def draw_bboxes(img_np, pred_mask, gt_mask, score_map=None, score_thresh=0.1):
    """
    Returns (vis_chw, fn_chw) both (C, H, W) uint8 for TensorBoard.
    vis_chw : GT boxes green, prediction boxes blue.
    fn_chw  : false-negative GT boxes in red, correctly-hit GT boxes in green.
    A GT region is a false negative when it has zero overlap with pred_mask.
    """
    vis    = (np.clip(img_np, 0, 1) * 255).astype(np.uint8).copy()
    fn_vis = vis.copy()

    # GT boxes – green on main image; blue=FN / green=hit on FN image
    gt_labeled = sk_measure.label(gt_mask.astype(np.uint8), connectivity=2)
    for region in sk_measure.regionprops(gt_labeled):
        r0, c0, r1, c1 = region.bbox
        cv2.rectangle(vis, (c0, r0), (c1, r1), (0, 255, 0), 1)
        is_fn = not pred_mask[gt_labeled == region.label].any()
        fn_color     = (0, 0, 255) if is_fn else (0, 255, 0)
        fn_thickness = 2           if is_fn else 1
        cv2.rectangle(fn_vis, (c0, r0), (c1, r1), fn_color, fn_thickness)
        if is_fn:
            cv2.putText(fn_vis, "FN", (c0, max(r0 - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Prediction boxes – blue on main image only
    pred_labeled = sk_measure.label(pred_mask.astype(np.uint8), connectivity=2)
    for region in sk_measure.regionprops(pred_labeled):
        r0, c0, r1, c1 = region.bbox
        score = float(score_map[(pred_labeled == region.label)].max()) if score_map is not None else 1.0
        if score < score_thresh:
            continue
        cv2.rectangle(vis, (c0, r0), (c1, r1), (30, 144, 255), 1)
        cv2.putText(vis, f"{score:.2f}", (c0, max(r0 - 2, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (30, 144, 255), 1)

    return vis.transpose(2, 0, 1), fn_vis.transpose(2, 0, 1)  # (C,H,W), (C,H,W)
    
