import argparse

import cv2
import torch
import random
import numpy as np
from datetime import datetime

from lib.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a dataset")
    parser.add_argument("--cfg",default="", help="Config file")
    parser.add_argument("--split",
                        choices=["train", "test", "val"],
                        default='test',
                        help="Dataset split to visualize")
    args = parser.parse_args()

    return args


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    args = parse_args()
    cfg = Config(args.cfg)
    while True:
        ret, frame_ = self.vcap.read()
        tp=[]
        tp.append(frame_)
        train_dataset = cfg.get_dataset(args.split)
        for idx in range(len(frame_)):
            img, _, _ = train_dataset.draw_annotation(idx)
            cv2.imwrite('images/'+str(datetime.today())+str(cfg)+'.jpg', img)
            #cv2.imshow('sample', img)
            cv2.waitKey(0)
        cv2.imshow("VideoFrame", img)
        if cv2.waitKey(1) == 27:
            vcap.release()  # 메모리 해제
            cv2.destroyAllWindows()  # 모든창 제거, 특정 창만듣을 경우 ("VideoFrame")
            break;


if __name__ == "__main__":
    main()
