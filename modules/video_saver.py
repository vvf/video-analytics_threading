import cv2
import os
import logging
from datetime import datetime

from modules.main import get_date_dirname
from threading import Thread

logger = logging.getLogger(__name__)


class ViewSaver:
    def __init__(self, dims):
        self.video_saver = None
        self.file_name = None
        if isinstance(dims, list):
            dims = tuple(dims)
        self.dims = dims
        self.queue = []
        self.worker_thread = None

        self.temp_filename = 'current_{:%Y%m%d_%H%M%S}.mp4'.format(datetime.now())
        self.video_saver = cv2.VideoWriter(self.temp_filename,
                                           cv2.VideoWriter_fourcc(*'MP4V'),
                                           15, tuple(reversed(self.dims)), 1)
        self.allow_add = True
        self.objects_types = set()

    def write_frame(self, image, objects_types=None):
        # if self.dims != tuple(image.shape[:2]):
        #     msg = "invalid size {} ".format(image.shape[:2])
        #     print("\n" + msg)
        #     logger.error(msg)
        #     return

        if objects_types:
            self.objects_types = self.objects_types.union(objects_types)

        if not self.allow_add:
            logger.debug("Add frame disabled. ")
            return
        self.queue.append(image)

        if not self.worker_thread:
            video_file_name = 'motion_{:%Y%m%d_%H%M%S}.mp4'.format(datetime.now())
            self.file_name = os.path.join(get_date_dirname(), video_file_name)
            self.worker_thread = Thread(target=self.write_worker, name="View_file_writer", daemon=True)
            self.worker_thread.start()

    def write_worker(self):
        print("\nStart save to file - {}\n".format(self.file_name))
        logger.info(f'Start save to file - {self.file_name}')
        while self.allow_add or self.queue:
            if not self.queue:
                continue

            # logger.info("Write video frame of {}".format(len(self.queue)))
            image = self.queue.pop(0)
            self.video_saver.write(image)

        logger.info("Closing video file, move to {}".format(self.file_name))
        self.video_saver.release()
        self.video_saver = None
        # TODO: check length of file and
        if self.objects_types:
            os.rename(self.temp_filename, self.file_name)
        else:
            os.unlink(self.temp_filename)
        self.file_name = None

    def close_video(self):
        if self.video_saver:
            self.allow_add = False
            if not self.worker_thread and self.temp_filename:
                self.video_saver.release()
                logger.info(f"Don't save current video.")
                os.unlink(self.temp_filename)

    def fill_queue(self, queue):
        # write all wat was before motion (without looking)
        pre_image = queue.pop()
        self.queue += queue
        self.write_frame(pre_image)
        queue.clear()
