from threading import Thread

import time
import logging
from datetime import datetime

import cv2
import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RTSPReaderThread(Thread):
    instance_counter = 0

    def __init__(self, *args, **kwargs):
        self.is_new_frame = False
        self.working = True
        self.vcap = None
        self.last_frame = None
        self.instance_no = RTSPReaderThread.instance_counter
        self.rtsp_url = kwargs.pop('rtsp_url', settings.rtsp_url)
        kwargs['name'] = f'RTSPCamReader-{RTSPReaderThread.instance_counter}'
        RTSPReaderThread.instance_counter += 1
        super(RTSPReaderThread, self).__init__(*args, **kwargs)

    def run(self):
        self.open_video()
        while self.working:
            self.read_frame()

    def stop(self):
        self.working = False


    def open_video(self):
        if self.vcap:
            self.vcap.release()
            self.vcap = None
            time.sleep(.5)

        self.vcap = cv2.VideoCapture(self.rtsp_url)

    def read_frame(self):
        ret = False
        cnt = 0
        last_ts = datetime.now()
        while not ret and self.working:
            ret, image = self.vcap.read()
            frame_time = datetime.now() - last_ts
            if not self.working:
                return
            if ret:
                self.last_frame = image
                self.is_new_frame = True
                return
            if frame_time.total_seconds() > 10:
                print('{}: no frames  {:3d}        \t\t'.format(self.instance_no, cnt), end='\r')
                cnt += 1
                if cnt > 200 or frame_time.total_seconds() > 30:
                    logger.error('No frame - reopen stream. cnt={}, sec={}'.format(cnt, frame_time.total_seconds()))
                    cnt = 0
                    self.open_video()
                time.sleep(0.01)
            last_ts = datetime.now()
