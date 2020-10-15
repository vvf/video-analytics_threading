import os
import logging
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

logger = logging.getLogger(__name__)


class NetProcessingImageBelt:
    next_model = None
    net_file: str = ''

    def __init__(self, ie: IECore, default_inputs=None):
        self.default_inputs = default_inputs or {}
        self.queue = []
        self.last_saved_frame_no = None
        max_num = ie.get_metric("MYRIAD", 'RANGE_FOR_ASYNC_INFER_REQUESTS')[1]
        self.max_num = max_num
        self.free_request_ids = list(range(max_num))
        if self.net_file.endswith('.xml') or self.net_file.endswith('.bin'):
            self.net_file = os.path.splitext(self.net_file)[0]
        logger.info(f"Load net to device - {os.path.basename(self.net_file)}")
        self.net = ie.read_network(model=self.net_file + '.xml', weights=self.net_file + '.bin')
        self.input_layer_name = self.get_input_layer_name()
        self.output_layer_name = next(iter(self.net.outputs.keys()))
        input_shape = self.net.input_info[self.input_layer_name].input_data.shape
        self.INPUT_SHAPE = tuple(input_shape)
        n, *input_shape = input_shape
        self.NET_INPUT_IMAGE_SIZE = tuple(reversed(input_shape[-2:]))
        logger.info(f'INPUT_SHAPE={input_shape}, NET_INPUT_IMAGE_SIZE={self.NET_INPUT_IMAGE_SIZE}')
        logger.info("Loading IR to the plugin...")
        self.exec_net = ie.load_network(network=self.net, num_requests=self.max_num, device_name="MYRIAD")

    def get_input_layer_name(self):
        for blob_name in self.net.input_info:
            if len(self.net.input_info[blob_name].input_data.shape) == 4:
                return blob_name

    def enqueue(self, image, frame_no=0, *args):
        while len(self.queue) >= self.max_num:
            self.wait_of_freeing()
        if not self.free_request_ids:
            return
        request_id = self.free_request_ids.pop(0)
        logger.info(f'input image ({frame_no}) shape - {image.shape}')
        in_frame = self.make_blob_from_image(image)
        logger.info(f"Start request #{request_id} ({frame_no})")
        self.queue.append((request_id, image, in_frame, frame_no) + tuple(args))
        self.exec_net.start_async(request_id=request_id, inputs=self.make_inputs(in_frame))

    def make_blob_from_image(self, image):
        in_frame = cv2.resize(image, self.NET_INPUT_IMAGE_SIZE)
        shape_before_transpose = in_frame.shape
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        logger.info(
            f'in_frame shape {shape_before_transpose} >transpose=> {in_frame.shape} >reshape=> {self.INPUT_SHAPE}')
        in_frame = in_frame.reshape((1,) + self.INPUT_SHAPE)
        return np.expand_dims(in_frame, axis=0)

    def wait_of_freeing(self):
        request_id, image, in_frame, frame_no, *args = self.queue.pop(0)
        if self.exec_net.requests[request_id].wait(-1) == 0:
            self.process_request_result(request_id, image, frame_no, *args)
            logger.info(f"Free request  #{request_id}")
            self.free_request_ids.append(request_id)
        else:
            logger.error(f"Waiting request {request_id} not return zero")

    def process_request_result(self, request_id, image, frame_no, *args):
        detections = self.exec_net.requests[request_id].output_blobs[self.output_layer_name]
        labels = self.add_detections(image, detections, frame_no, *args)
        self.output(image, frame_no, labels, *args)

    def add_detections(self, image, detections, frame_no, *args):
        return []

    def make_inputs(self, in_frame):
        return {
            **self.default_inputs,
            self.input_layer_name: in_frame,
        }

    def output(self, image, frame_no, labels, *args):
        if self.next_model and callable(getattr(self.next_model, 'enqueue', None)):
            args += (labels,)
            self.next_model.enqueue(image, frame_no, *args)


class LPRDetectionsParseMixin:
    model_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                     '<Anhui>',
                     '<Beijing>', '<Chongqing>', '<Fujian>', '<Gansu>', '<Guangdong>', '<Guangxi>', '<Guizhou>',
                     '<Hainan>',
                     '<Hebei>', '<Heilongjiang>', '<Henan>', '<HongKong>', '<Hubei>', '<Hunan>', '<InnerMongolia>',
                     '<Jiangsu>', '<Jiangxi>', '<Jilin>', '<Liaoning>', '<Macau>', '<Ningxia>', '<Qinghai>',
                     '<Shaanxi>',
                     '<Shandong>', '<Shanghai>', '<Shanxi>', '<Sichuan>', '<Tianjin>', '<Tibet>', '<Xinjiang>',
                     '<Yunnan>',
                     '<Zhejiang>', '<police>',
                     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                     'U', 'V', 'W', 'X', 'Y', 'Z']

    def add_detections(self, image, detections, frame_no):
        frame_h, frame_w = image.shape[:2]
        # logger.info(f'{frame_no}\tdetection shape - {detections.buffer.shape}')
        text = ''
        for i in detections.buffer.reshape((88,)).astype('int8'):
            if i < 0:
                continue
            c = self.model_classes[i]
            if c.startswith('<'):
                c = ''
            text += c
        if not text:
            return ''
        logger.info(f'{frame_no} => {text}')
        return text


class ImageBoxesInDetectionsMixin:
    label_names = 'person-vehicle-bike'.split('-')
    margin = 0
    min_score = 0.4
    colors = [(23, 23, 150), (200, 23, 200), (200, 230, 50)]

    def add_detections(self, image, detections, frame_no, *args):
        labels = set()

        for detection in detections.buffer[0, 0, :, :]:
            image_id, label, score, x_min, y_min, x_max, y_max = detection
            if score < self.min_score:
                continue
            label = int(label)
            det_label = self.label_names[label - 1]
            labels.add(det_label)
            self.outline_object_on_image(image, detection, det_label)
        return labels

    def outline_object_on_image(self, image, detection, det_label=None):
        image_id, label, score, x_min, y_min, x_max, y_max = detection
        frame_h, frame_w = image.shape[:2]
        left = int(x_min * frame_w) + self.margin
        if left < 0:
            left = 0
        top = int(y_min * frame_h)
        if top < 0:
            top = 0
        right = int(x_max * frame_w) + self.margin
        bottom = int(y_max * frame_h)
        label = int(label)
        color = self.colors[label] if label < len(self.colors) else (23, 230, 210)
        cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), color, thickness=2)

        cv2.putText(image, det_label + ' ' + str(round(score * 100, 1)) + ' %', (left - 5, top - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
        return (int(left), int(top)), (int(right), int(bottom))
