#!/usr/bin/env python3

import cv2
import numpy as np
from datetime import datetime
import time
import os
import sys
import logging

from settings import rtsp_url

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

vcap = None
first_image_small = None
detectors = {}
look_at_image_thread = None

def open_video():
    global vcap
    if vcap:
        vcap.release()
        vcap = None
        time.sleep(1)

    vcap = cv2.VideoCapture(rtsp_url)

BLUR_PARAM = (11,11)

SMALL_IMG_DIM = (480,270) #(576,324)
SMALL_IMG_K = SMALL_IMG_DIM[0]/1920


def read_first_image():
    global vcap, first_image, first_image_small

    first_image = read_frame()

    first_image_small = cv2.resize(first_image, SMALL_IMG_DIM)
    first_image_small = cv2.cvtColor(first_image_small, cv2.COLOR_BGR2GRAY)
    first_image_small = cv2.GaussianBlur(first_image_small, BLUR_PARAM, 0)
    logger.info('Readed first image')
    cv2.imwrite('image_small.png', first_image_small)


def read_frame():
    ret = False
    cnt = 0
    last_ts = datetime.now()
    while not ret:
        ret, image = vcap.read()
        frame_time = datetime.now() - last_ts
        if not ret or frame_time.total_seconds() > 10:
            print('no frames  {:3d}        \t\t'.format(cnt), end='\r')
            cnt += 1
            if cnt > 200 or frame_time.total_seconds() > 5:
                logger.error('No frame - reopen stream. cnt={}, sec={}'.format(cnt, frame_time.total_seconds()))
                cnt = 0
                open_video()
            time.sleep(0.01)
        last_ts = datetime.now()

    return image  #  [45:1045,900:1900]

def wait_motion(wait_no_motion=False, wait_frames=2):
    global first_image_small
    logger.info('wait {}motion'.format(wait_no_motion and ' NO' or ''))
    frame_no = 0
    fps = 0
    fps_cnt = 0
    last_s = None
    motion_frame_count = 0
    no_motion_frame_count = 0
    image_small = None
    video_saver = None
    frame_dims =  SMALL_IMG_DIM
    if wait_no_motion:
        image = read_frame()
        # frame_dims = tuple([d // 2 for d in image.shape[:2]])
        # video_h, video_w = frame_dims
        video_h, video_w = image.shape[:2]
#        video_file_name = 'appsrc ! autovideoconvert ! h264parse ! rtph264pay config-interval=10 pt=96 ! filesink location=motion_{:%Y%m%d_%H%M%S}.mp4 sync=true'.format(datetime.now())
        video_file_name = 'motion_{:%Y%m%d_%H%M%S}.mp4'.format(datetime.now())
        print("Start save video to {}: {}".format(video_file_name, frame_dims))
        video_saver = cv2.VideoWriter(video_file_name,
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                         15, (video_w, video_h), 1)
        video_saver.write(image)
        print("Written first frame")
    while True:
        image = read_frame()
        if no_motion_frame_count and video_saver:
            # image_save = cv2.resize(image, (video_w, video_h))
#            image_save = cv2.flip(image_save, 0)
            video_saver.write(image)
        image_small = cv2.resize(image, SMALL_IMG_DIM)
        image_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
        image_small = cv2.GaussianBlur(image_small, BLUR_PARAM, 0)
        delta = cv2.absdiff(first_image_small, image_small)
        thresh = cv2.threshold(delta, 35, 255, cv2.THRESH_BINARY)[1]
        now_ts = datetime.now()
        count_nonzero_thresh = np.count_nonzero(thresh)
        print('frame {:5d} {:2d} @{:%T}  nz:{:5d} {:3d} {:3d}       '.format(
            frame_no, fps,
            now_ts, count_nonzero_thresh, 
motion_frame_count, no_motion_frame_count
), end='\r')
        frame_no += 1
        fps_cnt +=1
        new_s = now_ts.strftime('%S')
        if new_s != last_s:
            last_s=new_s
            fps = fps_cnt
            fps_cnt = 0

        if count_nonzero_thresh > 250:   # more than square , all image 300x300 = 90000, so more than 1%
            no_motion_frame_count = 0
            if not wait_no_motion:
                motion_frame_count += 1
                if motion_frame_count > wait_frames:
                    break
        else:
            motion_frame_count = 0
            if wait_no_motion:
                no_motion_frame_count += 1
                if no_motion_frame_count > wait_frames: # frames with after no motion
                    break
        
#        if fps_cnt % 10 == 9:
#            first_image_small = image_small

        first_image_small = (first_image_small*0.75).astype('uint8') + (image_small*0.25).astype('uint8')

#    cy,cx = [o.min() + o.max() for o in np.where(thresh != 0)]
    if no_motion_frame_count:
        print("\nNO motion detected")
        logger.info('NO Motion detected')
        if video_saver:
            video_saver.release()
            video_saver = None
    else:
        print("\nmotion detected")
        logger.info('Motion detected')
        cv2.imwrite('delta.png', delta)
        cv2.imwrite('image_small.png', image_small)
        cv2.imwrite('thresh.png', thresh)
#    len_cntrs = 2*0.3
#    cx = int(cx/len_cntrs)
#    cy = int(cy/len_cntrs)
#    logger.info('Center at {}, {}'.format(cx, cy))
#    cx, cy = 1500, 0
#    if cx < 500:
#        cx = 500
#    if cy < 500:
#        cy = 500
#    if cy > 580:
#        cy = 580
#    if cx > 1420:
#        cx = 1420
#    return image[cy-500:cy+500,cx-500:cx+500]
    return image



def main_loop():
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)

    open_video()
    read_first_image()
    image = wait_motion(True, 250)
    return
    while True:
        image = wait_motion(False)
        image = wait_motion(True)


main_loop()
