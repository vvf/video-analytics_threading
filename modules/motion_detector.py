import cv2
import numpy as np
from datetime import datetime
import logging
from threading import Thread, Lock

from modules.rtsp_reader import RTSPReaderThread
from settings import SMALL_IMG_DIM, BLUR_PARAM, FRAMES_OF_REAL_MOTION, POST_MOTION_FRAMES, PRE_MOTION_FRAMES
from modules.main import writer, read_frame, close_video, look_image_queue

logger = logging.getLogger(__name__)


class MotionDetector:
    NO_MOTION = 0
    MOTION_START = 1
    MOTION_CONTINUE = 2
    MOTION_ENDING = 3

    def __init__(self, rtsp_reader: RTSPReaderThread, print_stat=True, cam_no=0, another_observers=None):
        self.no_motion_count = 0
        self.was_motion = False
        self.motion_frame_count = 0
        self.pre_motion_buffer = []
        self.rtsp_reader = rtsp_reader
        self.print_stat = print_stat
        self.first_image_small = None
        self.count_nonzero_thresh = 0
        self.cam_no = cam_no
        self.motion_threshold_continue = 50
        self.motion_threshold_start = 200
        self.another_observers = another_observers or []
        self.last_image = None
        self.has_image_to_show_lock = None
        if another_observers:
            self.has_image_to_show_lock = Lock()
            self.observers_thread = Thread(
                target=self.observers_thread,
                name=f'Observer #{cam_no}',
                daemon=True)
            self.observers_thread.start()

    def observers_thread(self):
        frame_no = 0
        while True:
            if self.last_image is None:
                self.has_image_to_show_lock.acquire()
            if self.last_image is None:
                continue
            try:
                for observer in self.another_observers:
                    observer(self.cam_no, self.last_image, frame_no)
                frame_no += 1
            except Exception as err:
                logger.exception(err)

            self.last_image = None

    def wait_motion(self):
        logger.info(f'wait motion ({self.cam_no})')
        frame_no = 0
        fps = 0
        fps_cnt = 0
        last_s = None
        self.motion_frame_count = 0
        while True:
            image = read_frame(self.rtsp_reader, cam_no=self.cam_no)
            if self.another_observers:
                self.last_image = image.copy()
                if self.has_image_to_show_lock.locked():
                    self.has_image_to_show_lock.release()

            has_motion = self.is_motion_in_frame(image)
            now_ts = datetime.now()
            if self.print_stat:
                print(f'frame {frame_no:5d} {fps:2d} @{now_ts:%T}  nz:{self.count_nonzero_thresh:5d}  '
                      f'vwq:{writer and len(writer.queue) or -1:3d}   liq:{len(look_image_queue):3d}      ', end='\r')
            frame_no += 1
            fps_cnt += 1
            new_s = now_ts.strftime('%S')
            if new_s != last_s:
                last_s = new_s
                fps = fps_cnt
                fps_cnt = 0
            if has_motion:
                break
        return image, has_motion

    def is_motion_in_frame(self, image):
        image_small = cv2.resize(image, SMALL_IMG_DIM)[120:, :]
        image_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
        image_small = cv2.GaussianBlur(image_small, BLUR_PARAM, 0)
        if self.first_image_small is None:
            self.first_image_small = image_small
            return MotionDetector.NO_MOTION
        delta = cv2.absdiff(self.first_image_small, image_small)
        thresh = cv2.threshold(delta, 35, 255, cv2.THRESH_BINARY)[1]

        self.count_nonzero_thresh = \
            max([
                np.count_nonzero(thresh_slice)
                for thresh_slice_x in np.split(thresh, 4, 1)
                for thresh_slice in np.split(thresh_slice_x, 3, 0)
            ])

        self.first_image_small = (self.first_image_small * 0.75).astype('uint8') + (image_small * 0.25).astype('uint8')

        motion_threshold = (self.was_motion or self.motion_frame_count > 0) and self.motion_threshold_continue or self.motion_threshold_start
        if self.count_nonzero_thresh > motion_threshold:  # more than square , all image 300x300 = 90000, so more than 1%
            # frame has motion
            self.motion_frame_count += 1
            self.no_motion_count = 0
            if self.was_motion:
                # prev frame - was motion - so it is true motion
                if self.print_stat:
                    print("motion continues  \r")
                logger.debug(f'Motion continued {self.cam_no}')
                return MotionDetector.MOTION_CONTINUE

            if self.motion_frame_count > FRAMES_OF_REAL_MOTION:  # be sure if it real motion
                self.was_motion = True
                if self.print_stat:
                    print("\nNew motion detected ")
                logger.debug(f'New motion detected {self.cam_no}')
                return MotionDetector.MOTION_START
        else:
            self.motion_frame_count = 0
            self.no_motion_count += 1
            if self.no_motion_count > POST_MOTION_FRAMES and self.was_motion:
                logger.debug(f"No motion for 15 frames {self.cam_no}")
                self.was_motion = False
            if self.was_motion:
                if self.print_stat:
                    print("after motion frame no {}   \r".format(self.no_motion_count))
                logger.debug(f"no motion, but wait and say is motion exists {self.no_motion_count} ({self.cam_no})")
                return MotionDetector.MOTION_ENDING
        # no motion
        self.pre_motion_buffer.append(image)
        while len(self.pre_motion_buffer) > PRE_MOTION_FRAMES:
            self.pre_motion_buffer.pop(0)
        return MotionDetector.NO_MOTION
