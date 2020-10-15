from datetime import datetime
import logging
import cv2
import os
from imutils.video import FPS
from openvino.inference_engine import IENetwork, IECore
from settings import rtsp_url, rtsp_url1, home_dir
from modules.video_saver import ViewSaver

# src_video = '~/coding/cars/2020/10/01/h15/motion_20201001_155302.mp4'
# src_video = '~/coding/cars/2020/09/20/h15/motion_20200920_152345.mp4'
# src_video = '~/coding/cars/2020/09/25/h15/motion_20200925_151338.mp4'
# src_video = '~/coding/cars/2020/09/25/h15/motion_20200925_151338.mp4'
src_video = rtsp_url

# net_file = '~/openvino_models/ir/person-vehicle-bike-detection-crossroad-0078'
# net_file = '~/openvino_models/ir/pedestrian-and-vehicle-detector-adas-0001'
net2_file = os.path.join(home_dir, 'openvino_models/ir/face-detection-retail-0004')
net_file = os.path.join(home_dir, 'openvino_models/ir/person-vehicle-bike-detection-crossroad-1016')

logger = logging.getLogger(__name__)


def get_input_layer_name(net: IENetwork):
    for blob_name in net.input_info:
        if len(net.input_info[blob_name].input_data.shape) == 4:
            return blob_name


class WriterNext:
    def __init__(self, detector):
        self.detector = detector
        self.frame_no = 0

    def write_frame(self, image, labels):
        self.frame_no += 1
        self.detector.enqueue(image, self.frame_no)


class Detector:
    def __init__(self, ie: IECore, writer: ViewSaver, dims, net_file: str):
        self.start_saving_on_object = False
        self.started_saving = True
        self.colors = [(23, 23, 150), (200, 23, 200), (200, 230, 50)]
        self.label_names = 'person-vehicle-bike'.split('-')
        self.margin = 0
        self.queue = []
        self.dims = dims
        self.last_saved_frame_no = None
        max_num = ie.get_metric("MYRIAD", 'RANGE_FOR_ASYNC_INFER_REQUESTS')[1]
        self.max_num = max_num
        self.free_request_ids = list(range(max_num))
        self.writer = writer
        if net_file.endswith('.xml') or net_file.endswith('.bin'):
            net_file = os.path.splitext(net_file)[0]
        self.net = ie.read_network(model=net_file + '.xml', weights=net_file + '.bin')
        self.input_layer_name = get_input_layer_name(self.net)
        self.output_layer_name = next(iter(self.net.outputs.keys()))
        n, *INPUT_SHAPE = self.net.input_info[self.input_layer_name].input_data.shape
        self.INPUT_SHAPE = tuple(INPUT_SHAPE)
        self.NET_INPUT_IMAGE_SIZE = tuple(INPUT_SHAPE[-2:])
        logger.info(f'INPUT_SHAPE={INPUT_SHAPE}, NET_INPUT_IMAGE_SIZE={self.NET_INPUT_IMAGE_SIZE}')
        logger.info("Loading IR to the plugin...")
        self.exec_net = ie.load_network(network=self.net, num_requests=self.max_num, device_name="MYRIAD")

    def enqueue(self, image, frame_no):
        while len(self.queue) >= self.max_num:
            self.wait_of_freeing()
        if not self.free_request_ids:
            return
        request_id = self.free_request_ids.pop(0)
        in_frame = cv2.resize(image, self.NET_INPUT_IMAGE_SIZE)
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape(self.INPUT_SHAPE)
        logger.info(f"Start request #{request_id}")
        self.queue.append((
            request_id, image, in_frame, frame_no, None
        ))
        self.exec_net.start_async(request_id=request_id, inputs={self.input_layer_name: in_frame})

    def wait_of_freeing(self):
        request_id, image, in_frame, frame_no, _ = self.queue.pop(0)
        if self.exec_net.requests[request_id].wait(-1) == 0:
            self.process_request_result(request_id, image, in_frame, frame_no)
            logger.info(f"Free request  #{request_id}")
            self.free_request_ids.append(request_id)
        else:
            logger.error(f"Waiting request {request_id} not return zero")

    def process_request_result(self, request_id, image, in_frame, frame_no):
        if self.last_saved_frame_no is not None and self.last_saved_frame_no + 1 != frame_no:
            logger.warning(f"Invalid frame sequence should be {self.last_saved_frame_no + 1}, but got {frame_no}")
        self.last_saved_frame_no = frame_no

        detections = self.exec_net.requests[request_id].output_blobs[self.output_layer_name]
        labels = self.add_detections(image, detections, frame_no)
        if self.start_saving_on_object or self.started_saving or labels:
            if self.start_saving_on_object and not self.started_saving:
                logger.info("Started saving frames to video")
                self.started_saving = True
            logger.info(f"Write frame {frame_no} with {labels or '-'}")
            self.writer.write_frame(image, labels)

    def add_detections(self, image, detections, frame_no):
        frame_h, frame_w = self.dims
        labels = set()
        for detection in detections.buffer[0, 0, :, :]:
            image_id, label, score, x_min, y_min, x_max, y_max = detection
            if score < 0.35:
                continue
            label = int(label)
            det_label = self.label_names[label - 1]
            labels.add(det_label)
            left = int(x_min * frame_w) + self.margin
            if left < 0:
                left = 0
            top = int(y_min * frame_h)
            if top < 0:
                top = 0
            right = int(x_max * frame_w) + self.margin
            bottom = int(y_max * frame_h)
            color = self.colors[label] if label < len(self.colors) else (23, 230, 210)
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), color, thickness=2)

            cv2.putText(image, det_label + ' ' + str(round(score * 100, 1)) + ' %', (left, top - 7),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

            # logger.info(f"{frame_no} {label}, {image_id}, ({left}, {top}) - ({right}, {bottom})")
        return labels


class ViewSaverDummy:
    def __init__(self, next_detector: Detector):
        self.detector = next_detector
        self.frame_no = 0
        self.labels = set()

    def write_frame(self, image, labels):
        self.frame_no += 1
        self.labels |= set(labels)
        self.detector.enqueue(image, self.frame_no)


def main():
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    vr = cv2.VideoCapture(src_video)
    ret, frame = vr.read()
    image = frame
    frame_h, frame_w = image.shape[:2]
    # margin = (frame_w - frame_h) // 2
    logger.info(f"Capture image size - {frame_w}x{frame_h}")
    writer = ViewSaver((frame_h, frame_w), cam_no='test')
    writer.write_frame(image, {'person'})
    ie = IECore()
    detector2 = Detector(ie, writer, (frame_h, frame_w), net_file=net_file)
    writer2 = ViewSaverDummy(detector2)
    detector = Detector(ie, writer2, (frame_h, frame_w), net_file=net2_file)
    detector.colors = [(255, 255, 255), (200, 255, 255)]
    detector.label_names = ['face', 'face1']

    # detector.start_saving_on_object = True
    # detector.started_saving = False

    fps = FPS()
    logger.info("Start")
    fps.start()
    frame_no = 0
    last_time = datetime.now()
    first_time = last_time
    temperature = '-'
    try:
        while ret:
            frame_no += 1
            detector.enqueue(frame, frame_no)
            new_time = datetime.now()
            dt = new_time - first_time
            if frame_no % 10 == 1:
                temperature = ie.get_metric("MYRIAD", "DEVICE_THERMAL")
            print(
                f'{frame_no} {new_time - last_time} {dt} {dt.total_seconds() > 0 and frame_no / dt.total_seconds() or 0:7.3}'
                f' tº={temperature: 7.3}'
            )
            cv2.putText(frame, f'{frame_no} {dt}', (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (30, 30, 255), 1)
            last_time = new_time
            fps.update()
            ret, frame = vr.read()
    except KeyboardInterrupt:
        print("\nStopping...\n\n")
        while detector.queue:
            detector.wait_of_freeing()
            fps.update()
    fps.stop()
    print(f'fps={fps.fps()}')
    writer.close_video()
    if writer.worker_thread and writer.worker_thread.is_alive():
        print(f"Wait for saver done {len(writer.queue)}")
        writer.worker_thread.join()


if __name__ == '__main__':
    main()
