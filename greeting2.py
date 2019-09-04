#!/usr/bin/env python3

import cv2
import numpy as np
from datetime import datetime, timedelta
import time
import os
import logging
import face_recognition as face
from threading import Thread
from settings import rtsp_url, FACE_GREETINGS_IMAGES, FACE_GREETINGS_COMMAND

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

vcap = None
detectors = {}
look_at_image_thread = None
look_image_queue = []

faces_enc = tuple()
known_names = tuple()
imgs_counters_by_name = {}


def init_logging():
    from logging.handlers import TimedRotatingFileHandler
    import sys

    err_lh = logging.StreamHandler()
    err_lh.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
    err_lh.setFormatter(formatter)
    logger.addHandler(err_lh)
    fh = TimedRotatingFileHandler('greeting2.log', 'D', 1, 7)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)


def load_faces(image_file, face_indexes):
    image = face.load_image_file(image_file)
    loc = face.face_locations(image)
    enc = face.face_encodings(image, loc)
    return [enc[i] for i in face_indexes]


def load_known_faces():
    global faces_enc, known_names, imgs_counters_by_name
    known_names = tuple()
    faces_enc = []

    for ld, names in FACE_GREETINGS_IMAGES.items():
        img_file, *idx = ld
        faces_enc.append(load_faces(img_file, idx))
        known_names += names

    faces_enc = tuple(faces_enc)

    imgs_counters_by_name = {name: 0 for name in known_names}


def greeting(user):
    if user not in known_names:
        pass
        # beep()
    else:
        os.system(FACE_GREETINGS_COMMAND.format(user=user))


def open_video():
    ''' Open video stream or REopen if it already exists '''
    global vcap
    if vcap:
        vcap.release()
        vcap = None
        time.sleep(0.5)

    vcap = cv2.VideoCapture(rtsp_url)


def read_frame():
    ret = False
    cnt = 0
    last_ts = datetime.now()
    image = None
    if not vcap:
        open_video()
    frame_time = None
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
        if 1 / frame_time.total_seconds() > 26:
            ret = False
        last_ts = datetime.now()

    return image, frame_time  # [45:1045,900:1900]


def init_dnn():
    global net

    detectors['face'] = cv2.CascadeClassifier(os.path.join('../cars/cv_data', 'haarcascade_frontalface_default.xml'))
    detectors['face2'] = cv2.CascadeClassifier(os.path.join('../cars/cv_data', 'haarcascade_profileface.xml'))


def imgwriter(filename, data):
    cv2.imwrite(filename, data)


no_face_counter = 0


def look_for_faces(img, full_img):
    detector = detectors['face']
    rimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cols, rows = img.shape[:2]

    faces = detector.detectMultiScale(rimg, 1.1, 5, maxSize=(650, 650), minSize=(50, 50))
    if len(faces) <= 0:
        return False

    for i, (x, y, w, h) in enumerate(faces):
        x -= 25
        if x < 0:
            x = 0
        y -= 25
        if y < 0:
            y = 0
        h += 50
        w += 50
        # if x + w >= cols:
        #     w = cols - x
        # if y + h >= rows:
        #     h = rows - y
        queue_face(img[y:y + h, x:x + w])
    return True


def queue_face(face_img: np.ndarray):
    global look_at_image_thread
    look_image_queue.append(face_img.copy())
    if not look_at_image_thread or not look_at_image_thread.is_alive():
        look_at_image_thread = Thread(target=face_comparator_worker())
        look_at_image_thread.start()


last_seen = {}
last_seen_unknown = datetime(1999, 1, 1)
face_search_time = 0
unknown_face_counter = 0


def face_comparator_worker():
    while look_image_queue:
        face_comparator(look_image_queue.pop())


def face_comparator(face_img):
    global face_search_time, unknown_face_counter
    face_locs = face.face_locations(face_img)
    if not face_locs:
        logging.info("False detection.")
        return
    logging.info("Face location = {}".format(face_locs[0]))
    print("\nFace location = {}".format(face_locs[0]))
    new_time = datetime.now()
    for j, face_enc in enumerate(face.face_encodings(face_img, face_locs)):
        recognized_people = face.compare_faces(faces_enc, face_enc, tolerance=0.55)
        has_known = False
        for i, is_known in enumerate(recognized_people):
            if not is_known:
                continue
            has_known = True
            face_name = known_names[i]
            print('\nHello {}: {}\n\n'.format(face_name, face_locs))
            _t, _r, _b, _l = face_locs[j]
            last_face = face_img[_t:_b, _l:_r]
            cv2.imwrite('faces/{}_{:03d}.png'.format(face_name, imgs_counters_by_name.get(face_name, 0)),
                        last_face)
            imgs_counters_by_name[face_name] = imgs_counters_by_name.get(face_name, 0) + 1
            if new_time - last_seen.get(face_name, datetime(1999, 1, 1)) > timedelta(minutes=10):
                greeting(face_name)
            last_seen[face_name] = new_time

        if not has_known:
            print("Unknown face {} saved".format(unknown_face_counter))
            cv2.imwrite('faces/unknown_{:03d}.png'.format(unknown_face_counter), face_img)
            unknown_face_counter += 1
            continue
    face_search_time = (datetime.now() - new_time).total_seconds()


def main_loop():
    init_logging()
    init_dnn()
    load_known_faces()
    open_video()
    frame, frame_time = read_frame()
    small_img_dim = tuple([d // 2 for d in frame.shape[:2]])
    last_time = datetime.now()
    print("Start loop. image size={}".format(tuple(reversed(small_img_dim))))
    while True:
        frame, frame_time = read_frame()
        new_time = datetime.now()
        small_frame = cv2.resize(frame, small_img_dim)
        look_for_faces(small_frame, frame)
        print('{:012.10f} fps {:012.10f} fps  {:2d}  {:12.10f} '.format(
            1 / (new_time - last_time).total_seconds(),
            1 / frame_time.total_seconds(),
            len(look_image_queue),
            face_search_time
        ), end='\r')
        last_time = new_time


if __name__ == '__main__':
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nbreaked. Exiting...\n")
    finally:
        vcap.release()
        cv2.destroyAllWindows()
