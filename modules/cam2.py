import logging
from datetime import datetime
from threading import Thread

import cv2
import numpy as np
import os

from modules.main import read_frame
from modules.motion_detector import MotionDetector
from modules.rtsp_reader import RTSPReaderThread
from modules.tgbot import send_event_photo, send_event_video, has_subscribers
from modules.video_saver import ViewSaver
from watcher import another_observers, notify_web_pages

logger = logging.getLogger(__name__)


class Cam2VideoSaver(ViewSaver):
    SUBSCRIPTION_NAME = 'cam2'
    frames = 0

    def close_video(self):
        super(Cam2VideoSaver, self).close_video()
        self.frames = 0

    def write_frame(self, image, objects_types=None):
        super(Cam2VideoSaver, self).write_frame(image, objects_types=objects_types or {'NO-LOOKED'})
        self.frames += 1

    def after_file_save(self):
        send_event_video(
            self.SUBSCRIPTION_NAME, self.file_name,
            caption=f"{os.path.basename(self.file_name)} frames={self.frames}"
        )


video_saver = None


def send_admin_motion_photo(image, motion_no='-', frame_no='-', count_th='-', msg='', thresh=None):
    image = image.copy()
    send_event_photo(
        image, 'cam2',
        caption=f'Motion #{motion_no} frame {frame_no}, th={count_th} {msg}'
    )
    if has_subscribers('cam2hl') and thresh is not None and thresh.max() > 210:
        image_slice = image[:, :20 - image.shape[1] // 2]
        corn_shape = (image.shape[0], image.shape[1] // 2)
        th = cv2.resize(thresh, tuple(reversed(corn_shape)))
        th = np.hstack((th, np.zeros((th.shape[0], 20), dtype='uint8')))
        th = cv2.GaussianBlur(th, (21, 21), 0)
        image_slice[(th > 0) & (th < 100), :] = (75, 75, 255)
        # image_slice = np.select([(th > 0) & (th < 200)], [image_slice * (0.7, 0.7, 0.5) + (0.3, 0.3, 0.5)], default=image_slice)
        # image_slice[(th > 0) & (th < 200), :] = (image_slice * (0.7, 0.7, 0.5) + (0.3, 0.3, 0.5)).astype('uint8')  # BGR
        send_event_photo(
            image, 'cam2hl',
            caption=f'Motion #{motion_no} frame {frame_no}, th={count_th} {msg}'
        )


def send_motion_image_bg(*args):
    frame_no = args[2]
    if frame_no > 5 and frame_no % 20 != 0:
        return
    logger.info(f"Send frame {args[1:]} to admin")
    send_admin_motion_photo_thread = Thread(
        target=send_admin_motion_photo,
        args=args,
        name='send_admin_motion_photo',
        kwargs={'thresh': last_thresh is not None and last_thresh.copy()}
    )
    send_admin_motion_photo_thread.start()


last_thresh = None


class Cam2MotionDetector(MotionDetector):
    def calc_number_of_nonzero(self, thresh: np.ndarray):
        global last_thresh
        last_thresh = thresh
        thresh[:, :thresh.shape[0] // 3] = 0
        # return np.count_nonzero(thresh)
        self.count_nonzero_thresh = max([
            np.count_nonzero(thresh_slice)
            for thresh_slice_x in np.split(thresh, 3, 1)
            for thresh_slice in np.split(thresh_slice_x, 3, 0)
        ])

    def make_small_image(self, image):
        image = super(Cam2MotionDetector, self).make_small_image(image)
        image[:, :140] = 0
        return image[:, :-image.shape[1] // 2]

    def blur(self, image_small):
        return cv2.GaussianBlur(image_small, (21, 21), 0)

    # def wait_motion(self):
    #     image, has_motion = super(Cam2MotionDetector, self).wait_motion()
    #     if has_motion == self.MOTION_ENDING:
    #         self.AVG_ALPHA = 0.2
    #     elif has_motion == self.MOTION_START:
    #         self.AVG_ALPHA = 0.1
    #     return image, has_motion

    def make_average_image(self, image_small):
        self.first_image_small = (self.first_image_small * (1-self.AVG_ALPHA)).astype('uint8') + \
                                 (image_small * self.AVG_ALPHA).astype('uint8')


def cam2_loop(show_messages):
    global video_saver
    import settings
    logger.info("Start CAM2 motion detector")
    rtsp_reader = RTSPReaderThread(rtsp_url=settings.rtsp_url2)
    rtsp_reader.start()
    image = read_frame(rtsp_reader, cam_no=2)
    logger.info(f"Read image resolution - {'x'.join(map(str,reversed(image.shape[:2])))}")
    video_saver = Cam2VideoSaver(dims=image.shape[:2], cam_no=2, fps=15)
    image = None
    motion_detector = Cam2MotionDetector(
        rtsp_reader, print_stat=show_messages, cam_no=2, another_observers=another_observers,
    )
    motion_detector.motion_threshold_continue = 300
    motion_detector.motion_threshold_start = 2000
    motion_id = 0
    motion_stopped = None
    frame_no = 0
    while True:
        image, motion_type = motion_detector.wait_motion()
        if motion_type == MotionDetector.MOTION_ENDING:
            video_saver.write_frame(image)
            if motion_stopped != motion_id:
                notify_web_pages(action='motionStop', camera=2, motionId=motion_id)
                motion_stopped = motion_id
                logger.info(f"CAM2: motion #{motion_id} end {frame_no}")
                print(f'\nCAM2: motion end {frame_no}\n')
                send_motion_image_bg(image, motion_id, 0, motion_detector.count_nonzero_thresh, 'END')
            frame_no += 1
            if motion_detector.no_motion_count == settings.POST_MOTION_FRAMES:
                video_saver.close_video()
                video_saver = Cam2VideoSaver(dims=video_saver.dims, cam_no=2, fps=15)
                frame_no = 0

        elif motion_type == MotionDetector.MOTION_START:
            motion_id += 1
            video_saver.fill_queue(motion_detector.pre_motion_buffer)
            video_saver.write_frame(image)
            send_motion_image_bg(image, motion_id, frame_no, motion_detector.count_nonzero_thresh, 'START')
            logger.info(f"CAM2: motion #{motion_id} start {frame_no}")
            frame_no += 1
            notify_web_pages(action='motionStart', camera=2, motionId=motion_id)
            print('\nCAM2: motion start\n')
            motion_stopped = None
        elif motion_type == MotionDetector.MOTION_CONTINUE:
            video_saver.write_frame(image)
            frame_no += 1
            send_motion_image_bg(image, motion_id, frame_no, motion_detector.count_nonzero_thresh, "CONTINUE")
            logger.info(f"CAM2: motion #{motion_id} re-start {frame_no}")

            notify_web_pages(action='motionStart', camera=2, motionId=motion_id)
            print('\nCAM2: motion ReStart\n')
            motion_stopped = None


if __name__ == '__main__':
    cam2_loop(True)
