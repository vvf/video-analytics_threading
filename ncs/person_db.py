#!/usr/bin/env python3
import re

import cv2
from datetime import datetime, timedelta
from redis import StrictRedis
import os
from settings import home_dir
from ncs.net_processing_belt import NetProcessingImageBelt, ImageBoxesInDetectionsMixin
from modules.tgbot import send_photo_to_admins
# from ..modules.tgbot import send_event_photo, bot
import logging
import numpy as np
from numpy.linalg import norm

from openvino.inference_engine import IENetwork, IECore

logger = logging.getLogger(__name__)

net_file = os.path.join(home_dir, 'openvino_models/ir/person-reidentification-retail-0267')
face_net_file = os.path.join(home_dir, 'openvino_models/ir/face-detection-0100')
face_vector_net_file = os.path.join(home_dir, 'openvino_models/ir/face-reidentification-retail-0095')
# face_vector_net_file = '/opt/intel/openvino_2020.4.287/deployment_tools/open_model_zoo/tools/downloader/public/face-recognition-mobilefacenet-arcface/FP16/face-recognition-mobilefacenet-arcface'
images_output_dir = os.path.join(home_dir, 'images')


class FaceFinder(NetProcessingImageBelt, ImageBoxesInDetectionsMixin):
    net_file = face_net_file
    label_names = ['face', 'face1', 'face2']
    min_score = 0.65

    def enqueue(self, image, frame_no='', *args):
        h, w = image.shape[:2]
        if w < 256 or h < 256:
            logger.info(f"image too small to search face - {w}x{h} ({os.path.basename(frame_no)})")
            return
        if h > w:
            if h * 2 // 3 > w:
                image = image[:w, :]
            else:
                margin = (h - w) // 2
                img = np.zeros((h, h, 3), dtype=image.dtype)
                img[:, margin:margin + w] = image
                image = img
        elif w > h:
            margin = (w - h) // 2
            image = image[margin:, :h]  # square from the middle

        super(FaceFinder, self).enqueue(image, frame_no, *args)

    def output(self, image, frame_no, labels, *args):
        pass

    def add_detections(self, image, detections, frame_no, *args):
        labels = set()
        FACE_WIDING_PX = 4
        for detection in detections.buffer[0, 0, :, :]:
            image_id, label, score, x_min, y_min, x_max, y_max = detection
            if score < self.min_score:
                continue
            label = int(label)
            det_label = self.label_names[label - 1]
            labels.add(det_label)
            box = self.outline_object_on_image(image, detection, det_label)
            (left, top) = [0 if ax < FACE_WIDING_PX else ax - FACE_WIDING_PX for ax in box[0]]
            (right, bottom) = [ax + FACE_WIDING_PX for ax in box[1]]

            object_image_slice = image[top:bottom, left:right]
            self.next_model.enqueue(object_image_slice, frame_no, image, *args)
        return labels


class NetReIdentification(NetProcessingImageBelt):
    net_file = net_file
    MINIMUM_SIMILARITY = 0.43
    subject = 'Person'
    output_dir = os.path.join(images_output_dir, 'pers')
    BASE_KEY = 'known_persons'

    def __init__(self, redis: StrictRedis, *args, **kwargs):
        self.redis = redis
        self.npp = 0
        self.known_persons = {
            npp.decode(): {
                'id': np.frombuffer(big_id, np.float32),
                'npp': 0
            }
            for npp, big_id in sorted(
                redis.hgetall(f'{self.BASE_KEY}_ids').items(),
                key=lambda item: int(item[0])
            )
        }
        for npp, person_name in redis.hgetall(f'{self.BASE_KEY}_names').items():
            self.known_persons[npp.decode()]['name'] = person_name.decode()
        self.known_persons_keys = list(self.known_persons.keys())
        self._persons_ids_mx = np.stack([
            p['id'] for p in self.known_persons.values()
        ], axis=0)
        self._norms = np.linalg.norm(self._persons_ids_mx, axis=1)
        super(NetReIdentification, self).__init__(*args, **kwargs)

    def add_detections(self, image: np.ndarray, detections, frame_no, *args):
        new_person_big_id = detections.buffer.reshape((max(detections.buffer.shape),))
        file_name = os.path.basename(frame_no)
        logger.info(f'Search similar person for {frame_no}')
        person_no, similarity = self.search_person(new_person_big_id)
        if person_no is not None:
            self.npp += 1
            self.known_persons[person_no]["npp"] += 1
            person_no = str(person_no)
            logger.info(f"found person {self.known_persons[person_no]['name']}")
            self.save_person_image(frame_no, image, f'{int(person_no):03}',
                                   f'{self.known_persons[person_no]["npp"]:03}_{file_name[4:]}',
                                   *args)
            return person_no
        person_no = str(len(self.known_persons))
        person_name = f'{self.subject} #{person_no}'

        self.known_persons[person_no] = {
            'name': person_name,
            'id': new_person_big_id,
            'npp': 0
        }
        self._norms = np.hstack((self._norms, norm(new_person_big_id)))
        self._persons_ids_mx = np.vstack((self._persons_ids_mx, new_person_big_id))
        self.known_persons_keys.append(person_no)
        logger.info(f"Add new {self.subject} #{person_no}  - {person_name}")
        self.redis.hset(f'{self.BASE_KEY}_names', person_no, person_name)
        self.redis.hset(f'{self.BASE_KEY}_ids', person_no, new_person_big_id.tobytes())
        self.save_person_image(frame_no, image, f'{int(person_no):03}', f'BASE-{file_name[4:]}', *args)
        return person_no

    def save_person_image(self, frame_no, image, person_no, file_name, *args):
        person_dir = os.path.join(self.output_dir, person_no)
        logger.info(f'Write file - {person_dir}/{file_name}')
        if not os.path.exists(person_dir):
            try:
                os.mkdir(person_dir)
            except:
                pass
        cv2.imwrite(os.path.join(person_dir, file_name), image)

    def search_person(self, new_person_id):
        if not self.known_persons:
            return None, None
        most_similar_person_no, similarity = self.find_max_similar(new_person_id)
        logger.info(f'Most similar person #{most_similar_person_no} with similarity={similarity}')
        if similarity < self.MINIMUM_SIMILARITY:
            return None, similarity
        return most_similar_person_no, similarity

    def find_max_similar(self, new_person_id):
        similarities = np.dot(self._persons_ids_mx, new_person_id) / (
                self._norms * norm(new_person_id))
        res = similarities.argmax()
        return self.known_persons_keys[res], similarities[res]


class FaceReIdentification(NetReIdentification):
    net_file = face_vector_net_file
    output_dir = os.path.join(images_output_dir, 'faces')
    BASE_KEY = 'known_faces'
    subject = 'Face'
    MINIMUM_SIMILARITY = 0.38

    def enqueue(self, image, frame_no='', *args):
        h, w = image.shape[:2]
        inp_w, inp_h = self.NET_INPUT_IMAGE_SIZE
        if w < inp_w // 2 or h < inp_h // 2:
            logger.info(f"image of face too small to compare face - {w}x{h} ({os.path.basename(frame_no)})")
            return

        super(FaceReIdentification, self).enqueue(image, frame_no, *args)

    def save_person_image(self, frame_no, image, face_no, file_name, *args):
        original_image = args[0] if args else image
        if len(args) >= 2:
            person_no = args[1]
            person_dir = os.path.join(self.output_dir, f'{int(person_no):03}')
            if not os.path.exists(person_dir):
                os.mkdir(person_dir)
            face_no = os.path.join(f'{int(person_no):003}', f'{int(face_no):03}')

        super(FaceReIdentification, self).save_person_image(
            frame_no,
            original_image,
            face_no, file_name,
            *args
        )


queue = os.environ.get('PERSON_IMAGES_QUEUE', 'persons')
redis = StrictRedis()


def enqueue_person_image_file(nets, image_path):
    image = cv2.imread(image_path)
    if image is None:
        logger.info(f'Can`t read {image_path}')
        return
    h, w = image.shape[:2]
    if w < 94:
        logger.info(f'\t{image_path} too small')
        return
    for net in nets:
        net.enqueue(image, image_path)


def main():
    logger = logging.getLogger('')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    ie = IECore()
    rec: NetReIdentification = NetReIdentification(redis, ie)
    find_face: FaceFinder = FaceFinder(ie)
    rec.next_model = find_face
    face_reid: FaceReIdentification = FaceReIdentification(redis, ie)
    find_face.next_model = face_reid

    try:
        os.mkdir(images_output_dir)
    except:
        pass
    try:
        os.mkdir(rec.output_dir)
    except:
        pass
    try:
        os.mkdir(face_reid.output_dir)
    except:
        pass
    logger.info(f"Wait for filenames. queue len at start -{redis.llen(queue)}\n\n")
    npp = 0
    begin_time = None
    try:
        while True:
            _, msg = redis.blpop(queue)
            if not begin_time:
                begin_time = datetime.now()
            filename = msg.decode()
            if filename:
                start = datetime.now()
                npp += 1
                enqueue_person_image_file([rec], filename)
                logger.info(f'queue len = {redis.llen(queue)} . frame - {datetime.now() - start}')

            while not redis.llen(queue) and rec.queue:
                rec.wait_of_freeing()
            while not redis.llen(queue) and find_face.queue:
                find_face.wait_of_freeing()
            while not redis.llen(queue) and face_reid.queue:
                face_reid.wait_of_freeing()

            if not redis.llen(queue):
                end_time = datetime.now()
                if npp:
                    logger.info(f'>>> Batch done! Count - {npp} fps: {npp / (end_time - begin_time).total_seconds()}')
                begin_time = None
                npp = 0

    except KeyboardInterrupt:
        pass
    while rec.queue:
        rec.wait_of_freeing()
    while find_face.queue:
        find_face.wait_of_freeing()
    while face_reid.queue:
        face_reid.wait_of_freeing()
    end_time = datetime.now()
    if npp:
        logger.info(f'count - {npp} fps: {npp / (end_time - begin_time).total_seconds()}')


if __name__ == '__main__':
    main()
