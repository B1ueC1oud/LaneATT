import pickle
import random
import logging

import cv2
import torch
import numpy as np
from tqdm import tqdm, trange
from imgaug.augmenters import Resize
import os
from natsort import natsorted
import re
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from torchvision.transforms import ToTensor
from lib.lane import Lane
from PIL import Image
from torchvision import transforms

from scipy.interpolate import InterpolatedUnivariateSpline
from torchvision import utils


# class Lane:
#     def __init__(self, points=None, invalid_value=-2., metadata=None):
#         super(Lane, self).__init__()
#         self.curr_iter = 0
#         self.points = points
#         self.invalid_value = invalid_value
#         self.function = InterpolatedUnivariateSpline(points[:, 1], points[:, 0], k=min(3, len(points) - 1))
#         self.min_y = points[:, 1].min() - 0.01
#         self.max_y = points[:, 1].max() + 0.01
#
#         self.metadata = metadata or {}
#
#     def __repr__(self):
#         return '[Lane]\n' + str(self.points) + '\n[/Lane]'
#
#     def __call__(self, lane_ys):
#         lane_xs = self.function(lane_ys)
#
#         lane_xs[(lane_ys < self.min_y) | (lane_ys > self.max_y)] = self.invalid_value
#         return lane_xs
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.curr_iter < len(self.points):
#             self.curr_iter += 1
#             return self.points[self.curr_iter - 1]
#         self.curr_iter = 0
#         raise StopIteration

class Runner:
    def __init__(self, cfg, exp, device, test_dataset, test_first_dir, test_second_dir, exp_name, hyper, hyper_param,
                 video_name, root_path, webcam=False, resume=False, view=None, deterministic=False):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.test_dataset = test_dataset
        self.test_first_dir = test_first_dir
        self.test_second_dir = test_second_dir
        self.logger = logging.getLogger(__name__)

        self.dataset_type = hyper_param[3]
        self.conf_threshold = hyper_param[0]
        self.nms_thres = hyper_param[1]
        self.nms_topk = hyper_param[2]

        self.root = root_path
        self.video_name = video_name
        self.hyper = hyper
        print(self.root)
        self.exp_name = "/{}/{}/".format(exp_name, self.hyper)
        self.name = test_first_dir + test_second_dir + test_dataset
        print(self.name)
        self.log_dir = self.name + self.exp_name  # os.path.join(self.name,self.exp_name)
        print(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        # Fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])
        if webcam:
            self.img_h = 360
            self.img_w = 640
            if webcam:
                self.vcap = cv2.VideoCapture(0)
                self.vcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 세로 사이즈
                self.vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)  # 가로 사이즈

            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        self.to_tensor = ToTensor()

    def _transform_annotations(self, idx):
        self.logger.info("Transforming annotations to the model's target format...")
        self.annotations = np.array(list(map(self.transform_annotation(idx), self._idx)))  # datasets
        self.logger.info('Done.')

    def train(self):
        self.exp.train_start_callback(self.cfg)
        starting_epoch = 1
        model = self.cfg.get_model()
        model = model.to(self.device)
        optimizer = self.cfg.get_optimizer(model.parameters())
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(model, optimizer, scheduler)
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg['epochs']
        train_loader = self.get_train_dataloader()
        loss_parameters = self.cfg.get_loss_parameters()
        for epoch in trange(starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs):
            self.exp.epoch_start_callback(epoch, max_epochs)
            model.eval()
            pbar = tqdm(train_loader)
            for i, (images, labels, _) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(images, **self.cfg.get_train_parameters())
                loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Scheduler step (iteration based)
                scheduler.step()

                # Log
                postfix_dict = {key: float(value) for key, value in loss_dict_i.items()}
                postfix_dict['lr'] = optimizer.param_groups[0]["lr"]
                self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), postfix_dict)
                postfix_dict['loss'] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)
            self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler)

            # Validate
            if (epoch + 1) % self.cfg['val_every'] == 0:
                self.eval(epoch, on_val=True)
        self.exp.train_end_callback()


    def webcam(self, epoch, on_val=False, save_predictions=False):
        # prediction_name="predictions_r34_culane"#
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info('Loading model %s', model_path)
        model.load_state_dict(self.exp.get_epoch_model(epoch))
        model = model.to(self.device)
        model.eval()
        # ret, frame = self.vcap.read()
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        self.exp.eval_start_callback(self.cfg)
        self.img_h = 360
        self.img_w = 640
        while True:
            self._idx = []
            self._idx.append({'lanes': [], 'path': ''})
            # self.dataset
            self.max_lanes = 3
            S = 72
            self.n_strips = S - 1
            self.n_offsets = S
            # self._transform_annotations(self._idx)
            self.strip_size = self.img_h / self.n_strips
            self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
            augmentations = []
            aug_chance = 0
            self.annotations = np.array(list(map(self.transform_annotation, self._idx)))
            self.img_h = 360
            self.img_w = 640

            transformations = iaa.Sequential([Resize({'height': self.img_h, 'width': self.img_w})])
            self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=aug_chance), transformations])
            ret, frame_ = self.vcap.read()
            frame_ori, _, _ = self.__getitem__(_idx=self.annotations, frame=frame_)
            _frame = frame_ori.cuda()
            _frame = torch.unsqueeze(_frame, dim=0)

            with torch.no_grad():
                idx=0
                output = model(_frame, **test_parameters)
                prediction = model.decode(output, as_lanes=True)
                predictions.extend(prediction)
                frame___ = torch.squeeze(_frame, dim=0)
                img = (frame___.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                # img=255-img
                img, fp, fn = self.draw_annotation(0, img=img, pred=prediction[0], frame=frame_)
            cv2.imshow("Webcam_laneATT", img)
            if cv2.waitKey(1) == 27:
                vcap.release()  # 메모리 해제
                cv2.destroyAllWindows()  # 모든창 제거, 특정 창만듣을 경우 ("VideoFrame")
                break;

    def label_to_lanes(self, label):
        # print("here")
        lanes = []
        for l in label:
            if l[1] == 0:
                continue
            xs = l[5:] / self.img_w
            ys = self.offsets_ys / self.img_h
            start = int(round(l[2] * self.n_strips))
            length = int(round(l[4]))
            xs = xs[start:start + length][::-1]
            ys = ys[start:start + length][::-1]
            xs = xs.reshape(-1, 1)
            ys = ys.reshape(-1, 1)
            points = np.hstack((xs, ys))
            # print(Lane(points=points))
            lanes.append(Lane(points=points))
        return lanes

    def draw_annotation(self, idx, label=None, pred=None, img=None, frame=None):
        # Get image if not provided
        # print(self.annotations)
        if True:
            _, label, _ = self.__getitem__(_idx=self.annotations, frame=frame)
            label = self.label_to_lanes(label)
        img = cv2.resize(img, (self.img_w, self.img_h))
        img_h, _, _ = img.shape
        # Pad image to visualize extrapolated predictions
        data = [(None, None, label)]
        if pred is not None:
            fp, fn, matches, accs = 0, 0, [1] * len(pred), [1] * len(pred)
            assert len(matches) == len(pred)
            data.append((matches, accs, pred))
        for matches, accs, datum in data:
            num = 0
            pad = 0
            temp = []
            for i, l in enumerate(datum):
                temp.append(l.points)
            if  len(datum) != 0:
                if len(datum) == 2:
                    if (sum(temp[0][:, 0])/len(temp[0][:, 0]))  > (sum(temp[1][:, 0])/len(temp[1][:, 0])) :
                        color=[(255,0,0),(0,0,255)]
                    else:
                        color = [(0, 0, 255), (255, 0, 0)]
                if len(datum) == 1:
                    #print(len(temp[0][0]))
                    if (sum(temp[0][:, 0])/len(temp[0][:, 0])) > 0.5:
                        color=[(255,0,0)]
                    else:
                        color = [(0, 0, 255)]
            for i, l in enumerate(datum):
                points = l.points
                # print(points)
                points[:, 0] *= img.shape[1]
                points[:, 1] *= img.shape[0]
                points = points.round().astype(int)
                points += pad
                xs, ys = points[:, 0], points[:, 1]
                for curr_p, next_p in zip(points[:-1], points[1:]):
                    img = cv2.line(img,
                                   tuple(curr_p),
                                   tuple(next_p),
                                   color=color[num],
                                   thickness=3 if matches is None else 3)
                num += 1

        return img, fp, fn

    def __getitem__(self, _idx=None, frame=None):
        item = _idx[0]
        img_org =frame #cv2.imread("/mnt/work/kim/KODAS1/Input/000397.jpg")
        line_strings_org = self.lane_to_linestrings(item['old_anno']['lanes'])
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)
        for i in range(30):
            img, line_strings = self.transform(image=img_org.copy(), line_strings=line_strings_org)
            line_strings.clip_out_of_image_()
            new_anno = {'path': item['path'], 'lanes': self.linestrings_to_lanes(line_strings)}
            try:
                label = self.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))['label']
                break
            except:
                if (i + 1) == 30:
                    self.logger.critical('Transform annotation failed 30 times :(')
                    exit()

        img = img / 255.
        img = self.to_tensor(img.astype(np.float32))
        return (img, label, 0)

    def eval(self, epoch, on_val=False, save_predictions=False):
        # prediction_name="predictions_r34_culane"#
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info('Loading model %s', model_path)
        model.load_state_dict(self.exp.get_epoch_model(epoch))
        model = model.to(self.device)
        model.eval()
        if on_val and self.test_dataset == None:
            dataloader = self.get_val_dataloader()
        elif self.test_dataset != None:
            dataloader = self.get_kodas_test_dataloader()
        else:
            dataloader = self.get_test_dataloader()
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        self.exp.eval_start_callback(self.cfg)
        with torch.no_grad():
            for idx, (images, _, _) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                #utils.save_image(images, "a.png")
                output = model(images, **test_parameters)
                prediction = model.decode(output, as_lanes=True)
                predictions.extend(prediction)
                if self.view:
                    img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0])
                    if self.view == 'mistakes' and fp == 0 and fn == 0:
                        continue

                    __name = self.log_dir + str(idx) + '.jpg'

                    cv2.imwrite(__name, img)
                    cv2.waitKey(0)
        image_folder = self.log_dir
        video_name = self.log_dir + self.video_name + '.avi'
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        images = natsorted(images)
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 30, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def transform_annotation(self, anno, img_wh=None):
        if img_wh is None:
            img_h = 360  # self.dataset.get_img_heigth(anno['path'])
            img_w = 640  # self.dataset.get_img_width(anno['path'])
        else:
            img_w, img_h = img_wh

        old_lanes = anno['lanes']
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates
        old_lanes = [[[x * self.img_w / float(img_w), y * self.img_h / float(img_h)] for x, y in lane]
                     for lane in old_lanes]
        # create tranformed annotations
        lanes = np.ones((self.max_lanes, 2 + 1 + 1 + 1 + self.n_offsets),
                        dtype=np.float32) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 length, S+1 coordinates
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            try:
                xs_outside_image, xs_inside_image = self.sample_lane(lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) == 0:
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips
            lanes[lane_idx, 3] = xs_inside_image[0]
            lanes[lane_idx, 4] = len(xs_inside_image)
            lanes[lane_idx, 5:5 + len(all_xs)] = all_xs

        new_anno = {'path': anno['path'], 'label': lanes, 'old_anno': anno}
        return new_anno

    def get_train_dataloader(self):
        train_dataset = self.cfg.get_dataset('train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.cfg['batch_size'],
                                                   shuffle=True,
                                                   num_workers=8,
                                                   worker_init_fn=self._worker_init_fn_)
        return train_loader

    def get_kodas_test_dataloader(self):
        self.cfg.set_kodas('test', self.dataset_type, self.conf_threshold, self.nms_thres, self.nms_topk, self.root)
        test_dataset = self.cfg.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader


    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset('val')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.cfg['batch_size'],
                                                 shuffle=False,
                                                 num_workers=8,
                                                 worker_init_fn=self._worker_init_fn_)
        return val_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
