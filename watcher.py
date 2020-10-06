#!/usr/bin/env python3
import json

import cv2
import logging
import numpy as np
import os
from datetime import datetime
from threading import Thread

from redis import Redis

import modules.main
from modules.alarm_zone import AlarmZone
from modules.main import read_frame, get_date_dirname, look_image_queue
from modules.motion_detector import MotionDetector
from modules import tgbot
from modules.rtsp_reader import RTSPReaderThread
from modules.video_saver import ViewSaver
from modules import webserver
from settings import SMALL_IMG_DIM, BLUR_PARAM, CLASSES, NOT_DRAW_CLASSES, CAR_IDXES, CAR_MIN_SQUARE, PERSON_IDXES, \
    PERSON_MIN_HEIGHT, EXACTLY_PERSON_INDEX, COLORS

logger = logging.getLogger('modules')
logger.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# from settings import *

detectors = {}
look_at_image_thread = None
alarmer: AlarmZone = None
motion_id = 0

redis = Redis()
# read_frame, writer, look_image_queue, close_video, get_date_dirname

rtsp_reader: RTSPReaderThread = None

end_motion_frames = []

EVENTS_CHANNEL = '/ws/control'

another_observers = [webserver.send_frame, tgbot.send_photo_if_need]


def notify_web_pages(**kwargs):
    redis.publish(EVENTS_CHANNEL, json.dumps(kwargs))


def read_first_image(rtsp_reader: RTSPReaderThread, cam_no=None):
    '''
    fill global variables first_image and first_image_small
    :return:
    '''
    global first_image

    first_image = read_frame(rtsp_reader, cam_no=cam_no)

    modules.main.first_image_small = cv2.resize(first_image, SMALL_IMG_DIM)
    modules.main.first_image_small = cv2.cvtColor(modules.main.first_image_small, cv2.COLOR_BGR2GRAY)
    modules.main.first_image_small = cv2.GaussianBlur(modules.main.first_image_small, BLUR_PARAM, 0)
    logger.info('Readed first image')
    # cv2.imwrite('image_small.png', first_image_small)


def init_dnn():
    global net

    detectors['license'] = cv2.CascadeClassifier('cv_data/haarcascade_russian_plate_number.xml')
    #    detectors['license'] = cv2.CascadeClassifier('cv_data/haarcascade_licence_plate_rus_16stages.xml')
    detectors['face'] = cv2.CascadeClassifier(os.path.join('cv_data', 'haarcascade_frontalface_default.xml'))
    detectors['face2'] = cv2.CascadeClassifier(os.path.join('cv_data', 'haarcascade_profileface.xml'))

    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
    logger.info('readed model')


def imgwriter(filename, data):
    cv2.imwrite(filename, data)


def image_parser_worker():
    is_odd = True
    while look_image_queue:
        queue_len = len(look_image_queue)
        if queue_len > 300:
            logger.warning(f"Image queue to long ({queue_len}), don't look to image")
            end_motion_frames.append(look_image_queue.pop(0))
        else:
            if is_odd or queue_len <= 50:
                look_at_image(look_image_queue.pop(0))
            else:
                logger.warning(f"Image queue to long ({queue_len}), skip every second image")
                end_motion_frames.append(look_image_queue.pop(0))
            is_odd = not is_odd
    logger.debug("finished image parser worker")
    if end_motion_frames:
        modules.main.writer.fill_queue(end_motion_frames)
        end_motion_frames.clear()
    if modules.main.writer.file_name:
        modules.main.close_video()


def look_at_image(image):
    # Thread(target=get_details_objects_plate, args=('face', image, date_dirname)).start()
    date_dirname = get_date_dirname()
    now_ts = datetime.now()
    (h, w) = image.shape[:2]
    margins = (w - h) // 2
    blob = cv2.dnn.blobFromImage(cv2.resize(image[:, margins: -margins], (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    # logger.info('start detecting')
    detections = net.forward()
    # logger.info('done detecting')

    # loop over the detections
    detected_classes = set()
    draw_boxes = []
    check_alarm_zone = []
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        image_for_cut = image.copy()
        if confidence > 0.6:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            detected_classes.add(CLASSES[idx])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX += margins
            endX += margins
            if idx in NOT_DRAW_CLASSES:
                continue
            dirname = CLASSES[idx]

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            logger.info('{} ({},{})-({},{}) {:%H%M%S}'.format(label, startX, startY, endX, endY, now_ts))
            draw_boxes.append((idx, startX, startY, endX, endY, label))
            if startY < 0:
                startY = 0
            if startX < 0:
                startX = 0

            fname = 'cut-{:%m%d_%H%M%S}-{}_{}.png'.format(now_ts, dirname, i)
            if not os.path.exists(os.path.join(date_dirname, fname)):
                Thread(target=imgwriter,
                       name='writer-' + fname,
                       args=(os.path.join(date_dirname, fname), image_for_cut[startY:endY, startX:endX])).start()

            startY -= 5
            startX -= 5
            if startY < 0:
                startY = 0
            if startX < 0:
                startX = 0

            if confidence > 0.7 and idx in CAR_IDXES and (endY - startY) * (
                    endX - startX) > CAR_MIN_SQUARE:  # square at least 300x300
                Thread(target=get_details_objects_plate,
                       args=('license', image[startY:endY, startX:endX], date_dirname, i)).start()
                check_alarm_zone.append(('car', len(draw_boxes) - 1))
            elif confidence > 0.7 and idx in PERSON_IDXES and (endY - startY) >= PERSON_MIN_HEIGHT:
                Thread(target=get_details_objects_plate,
                       args=('face', image[startY:endY, startX:endX], date_dirname, i, now_ts)).start()
                Thread(target=get_details_objects_plate,
                       args=('face2', image[startY:endY, startX:endX], date_dirname, i, now_ts)).start()
                if idx == EXACTLY_PERSON_INDEX:
                    check_alarm_zone.append(('person', len(draw_boxes) - 1))

    if not detected_classes:
        if modules.main.writer:
            modules.main.writer.write_frame(image)
        return

    for idx, startX, startY, endX, endY, label in draw_boxes:
        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    if alarmer and check_alarm_zone:
        for obj, box_no in check_alarm_zone:
            idx, startX, startY, endX, endY, label = draw_boxes[box_no]
            alarmer.check(startX, startY, endX, endY, obj == 'person', image, motion_id)

    # fname = '{:%m%d_%H%M%S}-{}.png'.format(datetime.now(), '_'.join(detected_classes))
    # logger.info('Write file {}'.format(fname))
    #    for dirname in sorted(detected_classes):
    #        if not os.path.exists(dirname):
    #            os.mkdir(dirname)
    # Thread(target=imgwriter, args=(os.path.join(date_dirname, fname), image)).start()
    if modules.main.writer:
        modules.main.writer.write_frame(image, detected_classes)


def get_details_objects_plate(obj_type, img, dirname, npp=0, now_ts=None):
    detector = detectors[obj_type]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not now_ts:
        now_ts = datetime.now()

    rotations = ((0, 1),)
    for rotate_n, rotationM in rotations:
        if rotate_n:
            logger.info("Try search {} plate in rotated image. {}".format(obj_type, rotate_n))
            rimg = cv2.warpAffine(gray, rotationM, (0, 0))
        else:
            rimg = gray

        plaques = detector.detectMultiScale(rimg, 1.2, 5, maxSize=(500, 500), minSize=(30, 30))
        if len(plaques) <= 0:
            logger.info('No {} detected, shape -{}'.format(obj_type, img.shape[:2]))
            continue

        for i, (x, y, w, h) in enumerate(plaques):
            fname = '{}-{:%m%d_%H%M%S}-{}-{}.png'.format(obj_type, now_ts, npp + i * 100, rotate_n)
            if os.path.exists(os.path.join(dirname, fname)):
                continue
            logger.info("Detected {}".format(obj_type))
            if rotate_n:
                rimg = cv2.warpAffine(img, rotationM, (0, 0))
            else:
                rimg = img
            cv2.imwrite(os.path.join(dirname, fname), rimg[y:y + h, x:x + w])
            if obj_type == 'license':
                logger.info(f"queue OCR license number {fname}")
                redis.lpush('licenses', os.path.join(dirname, fname))
        return True

    return False


def init_logging():
    from logging.handlers import TimedRotatingFileHandler
    import sys

    err_lh = logging.StreamHandler()
    err_lh.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(process)6d - %(threadName)s - %(levelname)s - %(message)s')
    err_lh.setFormatter(formatter)
    logger.addHandler(err_lh)
    fh = TimedRotatingFileHandler('watcher.log', 'D', 1, 7)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)
    logging.getLogger('modules').addHandler(fh)


look_at_image_thread_counter = 0

skip_next = False


def add_image_to_look(image):
    global look_at_image_thread, look_at_image_thread_counter, skip_next
    look_image_queue_len = len(look_image_queue)
    if look_image_queue_len > 200:
        print(f"Image queue to long ({look_image_queue_len}), skip every second image (when enq)")
        check_and_restart_image_parser_thread()
        return
    if look_image_queue_len > 50:
        skip_next = not skip_next
    else:
        skip_next = False
    if skip_next:
        print(f"Image queue to long ({look_image_queue_len}), skip every second image (when enq)")
    else:
        look_image_queue.append(image.copy())


def check_and_restart_image_parser_thread(is_checker=False):
    global look_at_image_thread, look_at_image_thread_counter
    if look_image_queue and (not look_at_image_thread or not look_at_image_thread.is_alive()):
        logger.info("starting image parser worker " + (look_at_image_thread and 'was thread' or ''))
        look_at_image_thread_counter += 1
        look_at_image_thread = Thread(target=image_parser_worker, name=f'image_parser-{look_at_image_thread_counter}')
        look_at_image_thread.start()


def main():
    init_logging()
    tgbot.start_bot()
    init_dnn()
    global alarmer, motion_id, rtsp_reader
    rtsp_reader = RTSPReaderThread()
    rtsp_reader.start()
    read_first_image(rtsp_reader, cam_no=0)
    alarmer = AlarmZone(*first_image.shape[:2])
    modules.main.writer = ViewSaver(first_image.shape[:2])
    motion_detector = MotionDetector(rtsp_reader, another_observers=another_observers,
                                     motion_filters={'image': lambda image: image[120:, :]})
    try:
        webserver.start_webserver(in_thread=True, daemon=True)
    except:
        exit(2)

    # https://stackoverflow.com/questions/22125256/python-multiprocessing-watch-a-process-and-restart-it-when-fails

    # from modules.cam1 import cam1_loop
    # Thread(target=cam1_loop, daemon=True, name="Cam1_Main").start()
    # from modules.cam2 import cam2_loop
    # Thread(target=cam2_loop, daemon=True, name="Cam2_Main").start()

    last_motion = None
    motion_stopped = None
    try:
        while True:
            image, motion_type = motion_detector.wait_motion()
            if motion_type == MotionDetector.MOTION_ENDING:
                # just add frame to result video
                if look_at_image_thread and look_at_image_thread.is_alive():
                    end_motion_frames.append(image)
                else:
                    modules.main.writer.write_frame(image)
                if motion_stopped != motion_id:
                    notify_web_pages(action='motionStop', camera=0, motionId=motion_id)
                    motion_stopped = motion_id
            elif motion_type == MotionDetector.MOTION_START:
                # write all what was before motion (without looking)
                modules.main.writer.fill_queue(motion_detector.pre_motion_buffer)
                motion_detector.pre_motion_buffer.clear()
                motion_id += 1
                notify_web_pages(action='motionStart', camera=0, motionId=motion_id)
            elif motion_type == MotionDetector.MOTION_CONTINUE:
                motion_stopped = None
                notify_web_pages(action='motionStart', camera=0, motionId=motion_id)
                if tgbot.count_of_users_waiting_motion() > 0:
                    Thread(target=tgbot.send_motion_start, args=(image.copy(), motion_id)).start()

            if motion_type != MotionDetector.MOTION_ENDING:
                # detect objects and save image with detected objects
                add_image_to_look(image)
            check_and_restart_image_parser_thread(True)
    except KeyboardInterrupt:
        webserver.close_translations()


# def test_videowriter():
#     init_logging()
#     open_video()
#     read_first_image()
#     writer = ViewSaver(first_image.shape[:2])
#     for j in range(5):
#         for i in range(150):
#             frame = read_frame()
#             modules.main.writer.write_frame(frame)
#         modules.main.writer.close_video()
#         modules.main.writer = ViewSaver(modules.main.writer.dims)
#         print("Write next file")
#         time.sleep(1)


if __name__ == '__main__':
    try:
        # test_videowriter()
        main()
    except Exception as err:
        logger.exception(err)
    except KeyboardInterrupt:
        tgbot.stop_bot()
        if rtsp_reader:
            rtsp_reader.stop()
            if rtsp_reader.is_alive():
                rtsp_reader.join(1000)
        import sys

        sys.exit(0)
    finally:
        if modules.main.writer:
            modules.main.writer.close_video()

    input("Press enter to continue")
