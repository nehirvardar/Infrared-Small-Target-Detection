# Basic module
from tqdm             import tqdm
from model.parse_args_test import  parse_args
import scipy.io as scio
import time
import csv
import os

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

import cv2
import json
import numpy as np

# Model
from model.model_DNANet import  Res_CBAM_block
#from model.model_ACM    import  ACM

from model.model_DNANet import  DNANet
from model.model_DNANet_vers1 import  DNANet_vers1
from model.model_DNANet_vers2 import  DNANet_vers2
from model.model_DNANet_vers3 import  DNANet_vers3

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.BBox  = COCOEvaluator(iou_thresh=0.1, score_thresh=0.1)
        self.PD_FA = PD_FA(1,10)
        self.mIoU  = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = "Maske"
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + args.dataset
            train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model == 'DNANet':
            from model.model_DNANet import DNANet, Res_CBAM_block
            model = DNANet(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block,
                           num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)

        elif args.model == 'DNANet_vers1':
            from model.model_DNANet_vers1 import DNANet_vers1, Res_CBAM_block as Res_CBAM_block_v1
            model = DNANet_vers1(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block_v1,
                                 num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)

        elif args.model == 'DNANet_vers2':
            from model.model_DNANet_vers2 import DNANet_vers2, Res_CBAM_block as Res_CBAM_block_v2
            model = DNANet_vers2(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block_v2,
                                 num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)

        elif args.model == 'DNANet_vers3':
            from model.model_DNANet_vers3 import DNANet_vers3, Res_CBAM_block as Res_CBAM_block_v2
            model = DNANet_vers3(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block_v2,
                                 num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)

        elif args.model == 'DNANet_vers4':
            from model.model_DNANet_vers4 import DNANet_vers4, Res_ECA_block
            model = DNANet_vers4(num_classes=1, input_channels=args.in_channels, block=Res_ECA_block,
                                 num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)

        elif args.model == 'DNANet_vers5':
            from model.model_DNANet_vers5 import DNANet_vers5, Res_CBAM_block as Res_CBAM_block_v5
            model = DNANet_vers5(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block_v5,
                                 num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)

        #elif args.model == 'ACM':
           # model       = ACM   (args.in_channels, layers=[args.blocks] * 3, fuse_mode=args.fuse_mode, tiny=False, classes=1)
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # ===================== Model Parameter Count =====================
        self.param_count = count_param(self.model)
        print(f"Model Parameters: {self.param_count:,}")

        # Evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Checkpoint
        try:
            checkpoint = torch.load(args.model_dir, weights_only=False)
            # Load trained model if checkpoint contains state_dict
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
                print(f"Loaded checkpoint: {args.model_dir}")
            else:
                print(f"Checkpoint loaded but no 'state_dict' key: {args.model_dir}")
        except Exception as e:
            print(f"Warning: could not load checkpoint '{args.model_dir}': {e}\nProceeding with randomly initialized weights.")

        target_image_path = 'mask_images_' + args.model
        target_dir = 'mask_images_concat'
        make_visulization_dir(target_image_path, target_dir)

        # ===================== TensorBoard Writer =====================
        tb_log_dir = os.path.join('runs', f'test_{args.model}_{args.dataset}')
        self.writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard log dizini: {tb_log_dir}")
        print(f"  --> tensorboard --logdir runs  komutuyla görüntüleyebilirsiniz.")

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()

        mean_IOU = 0.0

        # ===================== FPS Measurement =====================
        total_time = 0.0
        total_samples = 0

        with torch.no_grad():
            num = 0
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()

                # Measure inference time
                torch.cuda.synchronize()
                start_time = time.time()

                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred =preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)

                torch.cuda.synchronize()
                end_time = time.time()

                total_time += (end_time - start_time)
                total_samples += data.size(0)

                save_Pred_GT(pred, labels,target_image_path, val_img_ids, num, args.suffix)
                num += 1

                # If smoke flag set, stop after first batch
                if args.smoke:
                    print("Smoke mode: processed one batch, exiting test loop.")
                    break

                losses.    update(loss.item(), pred.size(0))
                start = i * args.test_batch_size
                batch_img_ids = val_img_ids[start : start + pred.size(0)]
                self.BBox.update(pred, labels, batch_img_ids)
                self.mIoU. update(pred, labels)
                self.PD_FA.update(pred, labels)

                # ---- TensorBoard BBox Görselleştirme (ilk 20 batch) ----
                if i < 20:
                    score_np   = torch.sigmoid(pred).cpu().numpy()   # (B,1,H,W)
                    pred_bin   = (pred > 0).cpu().numpy()             # (B,1,H,W)
                    labels_np  = labels.cpu().numpy()                 # (B,1,H,W)
                    data_np    = data.cpu().numpy()                   # (B,3,H,W)

                    for b in range(data.size(0)):
                        # Resmi denormalize et: (H,W,3) float [0,1]
                        img_np = data_np[b].transpose(1, 2, 0)
                        img_np = img_np * np.array([.229, .224, .225]) + np.array([.485, .456, .406])

                        # 2-boyutlu maskeleri çıkar
                        p_mask = pred_bin[b, 0]  if pred_bin.ndim  == 4 else pred_bin[b]
                        g_mask = labels_np[b, 0] if labels_np.ndim == 4 else labels_np[b]
                        s_map  = score_np[b, 0]  if score_np.ndim  == 4 else score_np[b]

                        vis_chw = draw_bboxes(img_np, p_mask, g_mask,
                                              score_map=s_map, score_thresh=0.1)

                        img_tag = f'BBox/{batch_img_ids[b]}'
                        self.writer.add_image(img_tag, vis_chw, global_step=0)

                _, mean_IOU = self.mIoU.get()
            FA, PD = self.PD_FA.get(len(val_img_ids))
            # bbox_recall, bbox_precision = self.BBox.get()
            test_loss = losses.avg
            scio.savemat(f'final_rapor_{args.model}_{args.dataset}.mat',
                         {'number_record1': FA, 'number_record2': PD})

            print('test_loss, %.4f' % (test_loss))
            print('mean_IOU:', mean_IOU)
            self.best_iou = mean_IOU

            # ===================== Compute Final Metrics (BBox-based as PRIMARY) =====================
            mAP_score, best_precision, best_recall = self.BBox.get()

            # FPS (frames per second) and it/s (iterations per second)
            fps = total_samples / total_time if total_time > 0 else 0.0
            its = len(self.test_data) / total_time if total_time > 0 else 0.0  # iterations per second

            print(f"\n===== BOUNDING BOX METRICS (PRIMARY - IoU=0.1) =====")
            print(f"BBox Best Recall:    {best_recall:.4f}")
            print(f"BBox Best Precision: {best_precision:.4f}")
            print(f"BBox mAP Score:      {mAP_score:.4f}")
            print(f"mIoU (Pixel-level):  {mean_IOU:.4f}")
            print(f"FPS:                 {fps:.2f}")
            print(f"it/s:                {its:.2f}")
            print(f"Parameters:          {self.param_count:,}")

            # ===================== Save CSV Report =====================
            csv_filename = f"test_report_{args.model}_{args.dataset}.csv"
            csv_path = os.path.join(os.getcwd(), csv_filename)

            file_exists = os.path.isfile(csv_path)
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow([
                        'Model',
                        'Dataset',
                        'Parameters',
                        'BBox_Precision',
                        'BBox_Recall',
                        'BBox_mAP_Score',
                        'mIoU',
                        'FPS',
                        'Test_Loss'
                    ])
                writer.writerow([
                    args.model,
                    args.dataset,
                    self.param_count,
                    f"{best_precision:.6f}",
                    f"{best_recall:.6f}",
                    f"{mAP_score:.6f}",
                    f"{mean_IOU:.6f}",
                    f"{fps:.2f}",
                    f"{test_loss:.6f}"
                ])

            print(f"\n>>> CSV report saved to: {csv_path}")

            bbox_sorted_recall = [best_recall]
            bbox_sorted_precision = [best_precision]

            save_result_for_test(dataset_dir, f'rapor_{args.model}', args.epochs, mAP_score,
                                 bbox_sorted_recall, bbox_sorted_precision, None, None, mAP_score)

            # Save COCO JSON format results using pycocotools
            self.BBox.save_final_json(dataset_dir, f'coco_results_{args.model}')
            self.BBox.save_pr_curve(dataset_dir, f'coco_results_{args.model}')

            # ===================== TensorBoard PR Eğrisi =====================
            if self.BBox._coco_eval is not None:
                prec_arr = self.BBox._coco_eval.eval['precision'][0, :, 0, 0, 2]
                rec_arr  = self.BBox._coco_eval.params.recThrs
                valid    = prec_arr > -1
                p_vals   = prec_arr[valid]
                r_vals   = rec_arr[valid]
                for idx in range(len(p_vals)):
                    self.writer.add_scalar(
                        'PR_Curve/Precision',
                        float(p_vals[idx]),
                        int(round(r_vals[idx] * 100))   # x ekseni: recall × 100  (0–100)
                    )

            # ===================== TensorBoard Scalar Metrikler =====================
            self.writer.add_scalar('Test/mAP@0.1',   mAP_score,      0)
            self.writer.add_scalar('Test/Precision',  best_precision, 0)
            self.writer.add_scalar('Test/Recall',     best_recall,    0)
            self.writer.add_scalar('Test/mIoU',       mean_IOU,       0)
            self.writer.add_scalar('Test/FPS',        fps,            0)
            self.writer.add_scalar('Test/Loss',       test_loss,      0)
            self.writer.close()
            print(f"\n>>> TensorBoard kayıtları: {tb_log_dir}")


            # =====================================================================
            # OLD VISUALIZATION CODE (commented out, kept for reference)
            # =====================================================================
            # source_image_path = dataset_dir + '\\images'
            # if args.mode == 'TXT':
            #     txt_path = test_txt
            #     ids = []
            #     with open(txt_path, 'r') as f:
            #         ids += [line.strip() for line in f.readlines()]
            #
            # for i in range(len(ids)):
            #     source_image = source_image_path + '\\' + ids[i] + args.suffix
            #     target_image = target_image_path + '\\' + ids[i] + args.suffix
            #     shutil.copy(source_image, target_image)
            # for i in range(len(ids)):
            #     source_image = target_image_path + '\\' + ids[i] + args.suffix
            #     img = Image.open(source_image)
            #     img = img.resize((256, 256), Image.ANTIALIAS)
            #     img.save(source_image)
            # for m in range(len(ids)):
            #     plt.rcParams['font.sans-serif'] = ['STSong']
            #     plt.figure(figsize=(10, 6))
            #     plt.subplot(1, 3, 1)
            #     img = plt.imread(target_image_path +'\\'+ ids[m] +args.suffix)
            #     plt.imshow(img,cmap = 'gray')
            #     plt.xlabel("Raw Image", size=11)
            #
            #     plt.subplot(1, 3, 2)
            #     img = plt.imread(target_image_path +'\\'+ ids[m] + '_GT'+args.suffix)
            #     plt.imshow(img,cmap = 'gray')
            #     plt.xlabel("Ground Truth", size=11)
            #
            #     plt.subplot(1, 3, 3)
            #     img = plt.imread(target_image_path +'\\'+ ids[m] + '_Pred'+args.suffix)
            #     plt.imshow(img,cmap = 'gray')
            #     plt.xlabel("Prediction", size=11)
            #
            #     plt.savefig(target_dir +'\\'+ ids[m].split('.')[0] + "_fuse"+args.suffix, facecolor='w', edgecolor='red')
            # =====================================================================


def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
