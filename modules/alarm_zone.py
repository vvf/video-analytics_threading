from time import sleep

from collections import defaultdict

import datetime
import logging
from threading import Thread

from modules import tgbot

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AlarmZone:
    def __init__(self, height, width):
        self.width = width
        self.height = height
        self.x1 = width // 5
        self.x2 = width // 2
        self.k = (self.x2 - self.x1) / height
        self.car_y = height // 4
        self.person_y = height // 2
        self.next_alarm_time = {}
        self.last_alarm_count = defaultdict(lambda: 0)
        self.last_motion_id = None
        self.skipped_images = 0
        self.wait_time = 0
        self.image = None
        self.wait_thread: Thread = None

    def is_in_zone(self, start_x, start_y, end_x, end_y, is_person_or_car):
        y1 = self.person_y if is_person_or_car else self.car_y
        return (start_x < self.x1 + self.k * start_y and start_y > y1) \
               or (start_x < self.x1 + self.k * end_y and end_y > y1)

    def check(self, start_x, start_y, end_x, end_y, is_person_or_car, image, motion_id=None):
        next_alarm_time = self.next_alarm_time.get(is_person_or_car)
        if self.last_motion_id != motion_id:
            self.reset_alarm_count(is_person_or_car)
            self.wait_thread = None  # if previous thread exists - it sends his last image
            self.last_motion_id = motion_id
        elif next_alarm_time and next_alarm_time > datetime.datetime.now():
            return

        if self.is_in_zone(start_x, start_y, end_x, end_y, is_person_or_car):
            obj = 'Человек' if is_person_or_car else 'Автомобиль'
            self.last_alarm_count[is_person_or_car] += 1
            last_alarm_count = self.last_alarm_count[is_person_or_car]
            logger.info(f"Alarm to {obj} count={last_alarm_count} motion_id={motion_id}")
            if not self.wait_thread or not self.wait_thread.is_alive():
                try:
                    self.wait_thread = WaitSendThread(
                        name=f"tgBotWaitSend-{motion_id}"
                    )
                    self.wait_thread.parent = self
                    self.wait_thread.start()
                    logger.info(f"Alarmer: Start wait and send image thread")
                except Exception as err:
                    logger.exception(err)
            self.wait_thread.set_image(image, f"{obj} в зоне мониторинга ({motion_id})", is_person_or_car)

    def reset_alarm_count(self, is_person_or_car=''):
        self.last_alarm_count[is_person_or_car] = 0

    def set_next_alarm_time(self, is_person_or_car=''):
        last_alarm_count = self.last_alarm_count[is_person_or_car]
        self.next_alarm_time[is_person_or_car] = datetime.datetime.now() + datetime.timedelta(
            seconds=30 + last_alarm_count * 10)
        logger.info(f"Alarmer: next after {self.next_alarm_time[is_person_or_car]}")


class WaitSendThread(Thread):
    def __init__(self, *args, **kwargs):
        super(WaitSendThread, self).__init__(*args, **kwargs)
        self.image = None
        self.title = None
        self.is_person_or_car = ''
        self.skipped_images = 0
        self.parent = kwargs.pop('parent', None)

    def set_image(self, image, title=None, is_person_or_car=''):
        self.image = image
        self.title = title
        self.is_person_or_car = is_person_or_car
        logger.info(f"Alarmer,Wait: Skip prev image. {self.skipped_images}")
        self.skipped_images += 1

    def run(self):
        try:
            logger.info("Alarmer,Wait: Start waiting")
            wait_time = 0
            while wait_time < 40:
                sleep(.05)
                wait_time += 1
                if self.skipped_images >= 7:
                    logger.info(f"Alarmer,Wait: Skipped {self.skipped_images}. stop waiting")
                    break
            logger.info(f"Alarmer,Wait: Waiting done {wait_time}")
            self.skipped_images = 0
            if self.parent:
                self.parent.wait_thread = None
                self.parent.set_next_alarm_time(self.is_person_or_car)
            if self.image is not None:
                tgbot.send_photo_to_admins(image=self.image, caption=self.title)
            else:
                logger.error("Alarmer,Wait: No image to send")
            self.image = None
        except Exception as err:
            logger.exception(err)