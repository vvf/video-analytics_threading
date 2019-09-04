import time

import cv2
import logging
import os
from datetime import datetime, timedelta

from modules.tgbot import bot

from settings import rtsp_url

logger = logging.getLogger(__name__)
writer = None
tg_notified_time = None

class LongReadFrameError(Exception):
    pass


def read_frame(rtsp_reader, long_wait_behaviour=0, wait_timeout=900, sleep_time=0.05, cam_no=None):
    global tg_notified_time
    i = 0
    wait_timeout = wait_timeout / sleep_time
    while not rtsp_reader.is_new_frame:
        time.sleep(sleep_time)
        i += 1
        if i > wait_timeout:
            break
    else:
        rtsp_reader.is_new_frame = False
        return rtsp_reader.last_frame.copy()

    if not tg_notified_time or (datetime.now() - tg_notified_time).total_seconds() > 1200:
        tg_notified_time = datetime.now()
        bot.message_to_admin(f"Camera {cam_no is None and '' or cam_no} looks like not worked. "
                             f"wait frame time was {timedelta(seconds=i*0.05)}")
    if long_wait_behaviour == 0:
        import sys
        sys.exit(10)
    elif long_wait_behaviour == 1:
        raise LongReadFrameError(f"Can't read frame long time! {timedelta(seconds=i*0.05)}")
    else:
        logger.error(f"Can't read frame long time {timedelta(seconds=i*0.05)}!")

def get_date_dirname(now_ts=None):
    if not now_ts:
        now_ts = datetime.now()
    images_paths = (
        now_ts.strftime('%Y'),
        now_ts.strftime('%m'),
        now_ts.strftime('%d'),
        now_ts.strftime('h%H'),
    )
    try:
        for i in range(len(images_paths)):
            p = os.path.join(*images_paths[:i + 1])
            if not os.path.exists(p):
                os.mkdir(p)
    except OSError as err:
        if 'Errno 28' in str(err):
            bot.message_to_admin("Диск переполнен!")
            os._exit(1)
    return os.path.join(*images_paths)


def close_video():
    global writer
    if writer.file_name:
        writer.close_video()
        from modules.video_saver import ViewSaver
        writer = ViewSaver(writer.dims)


vcap = None


def open_video():
    ''' Open video stream or REopen if it already exists '''
    global vcap
    if vcap:
        vcap.release()
        vcap = None
        time.sleep(1)

    vcap = cv2.VideoCapture(rtsp_url)


look_image_queue = []
first_image_small = None
