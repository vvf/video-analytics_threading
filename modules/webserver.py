import datetime
import json
import logging
from functools import partial
from http import server
from threading import Thread
from time import sleep

import cv2

logger = logging.getLogger(__name__)
WIDTH = 640
HEIGHT = 480

WATCHDOG_TIMEOUT = 25
FRAMES_TO_CLOSE = 500  # if 15 frames per second - so it will be


def watch_dog_loop(video_feeds):
    while True:
        ts = datetime.datetime.now()
        all_feeds = list(video_feeds.values())
        for feeds in all_feeds:
            for feed in feeds:
                if feed.start_time:
                    sec = (ts - feed.start_time).total_seconds()
                else:
                    sec = 0
                if sec > WATCHDOG_TIMEOUT:
                    logger.info(f"Close ({feed._stat()}) by watch dog")
                    print(f"Close ({feed._stat()}) by watch dog")
                    feed.stop_translation()
        sleep(1)


class VideoFeed:
    video_feeds = {}
    last_camera_jpg = {}
    camera_height = {}

    def __init__(self, response, cam_no):
        self.response = response
        self.cam_no = cam_no
        self.frames_sent = 0
        self.height = None
        self.start_time = datetime.datetime.now()
        self.sending_frame = False

    def write_frame(self, jpg, frame_no=None):
        # skip frames if not ready yet to send another frame
        if self.sending_frame:
            return True
        self.sending_frame = True
        try:
            self.response.wfile.write(b"--jpgboundary")
            self.response.send_header('Content-type', 'image/jpeg')
            self.response.send_header('Content-length', str(jpg.size))
            self.response.send_header('X-Timestamp', datetime.datetime.now().timestamp())
            self.response.send_header('X-Frame-No', frame_no or '0')
            self.response.end_headers()
            self.response.wfile.write(jpg)
            self.frames_sent += 1
            self.response.wfile.flush()
            if self.frames_sent >= FRAMES_TO_CLOSE:
                logger.info(f"{self._stat()}: translation done by count frames = {self.frames_sent}")
                self.stop_translation()
        except (BrokenPipeError, ConnectionResetError, OSError) as err:
            # logger.exception(err)
            logger.error("Ignore error, just stop translation")
            self.stop_translation()
        self.sending_frame = False
        # if frame_no:
        #     logger.debug(f'sent frame: {frame_no} to {self.response.client_address}')
        return True

    @classmethod
    def resize_img(cls, cam_no, img):
        height = cls.camera_height.get(cam_no)
        if not height:
            h, w = img.shape[:2]
            height = int(h * WIDTH / w)
            cls.camera_height[cam_no] = height
        return cv2.resize(img, (WIDTH, height))

    @classmethod
    def start_translation(cls, web_response, cam_no, only_last=True):
        if cam_no not in cls.video_feeds:
            cls.video_feeds[cam_no] = []
        new_feed = VideoFeed(web_response, cam_no)
        last_jpg = cls.last_camera_jpg.get(cam_no)
        if last_jpg is None:
            cls.video_feeds[cam_no].append(new_feed)
            return
        new_feed.write_frame(last_jpg)
        if only_last:
            new_feed.stop_translation()
            web_response.close_connection = True
            return
        cls.video_feeds[cam_no].append(new_feed)

    @classmethod
    def send_frame(cls, cam_no, img, frame_no=None):
        cam_listeners = VideoFeed.video_feeds.get(cam_no) or []
        if img.shape[1] != WIDTH:
            img = cls.resize_img(cam_no, img)
        success, jpg = cv2.imencode('.jpg', img)
        if not success:
            logger.error(f"Can't encode to jpeg {frame_no or ''}")
            return False
        cls.last_camera_jpg[cam_no] = jpg
        if cam_listeners:
            # logger.info(f"Send frame #{frame_no} to {len(cam_listeners)}")
            print(f"Send frame #{frame_no} to {len(cam_listeners)}     ", end='\r')
        for feed in cam_listeners:
            feed.write_frame(jpg, frame_no)

    def stop_translation(self):
        self.response._is_cam = False
        print(f"STOP serve video of camera #{self.cam_no} for {self.response.client_address}")
        try:
            VideoFeed.video_feeds[self.cam_no].remove(self)
        except (ValueError, KeyError):
            pass

        try:
            self.response.wfile.flush()
        except Exception as err:
            logger.exception(err)

        try:
            self.response.finish()
        except Exception as err:
            logger.exception(err)
        del self

    def _stat(self):
        return {
            'from': ':'.join(map(str, self.response.client_address)),
            'started': self.start_time and self.start_time.isoformat() or '',
            'camNo': self.cam_no
        }

    @classmethod
    def get_stat(cls):
        return {
            str(cam_no): [feed._stat() for feed in feeds]
            for cam_no, feeds in cls.video_feeds.items()
        }


def refresh_slides(time_to_sleep=None):
    import settings
    from pathlib import Path
    from redis import Redis
    if time_to_sleep:
        sleep(time_to_sleep)
    p = Path(settings.SLIDES_PATH)
    redis = Redis()
    imgs = [f.as_posix().replace(settings.SLIDES_PATH, '') for f in p.glob('**/*') if
            f.is_file() and not f.name.startswith('.') and '.AppleDouble' not in f.as_posix()]
    logger.info("Send list of {len(imgs)} images to redis channel /ws/control")
    redis.publish('/ws/control', json.dumps({'action': 'refreshSlides', 'slides': imgs}))


def close_translations():
    all_feeds = list(VideoFeed.video_feeds.values())
    for feeds in all_feeds:
        for feed in feeds:
            feed.stop_translation()


class MyResponder(server.SimpleHTTPRequestHandler):
    _is_cam = False

    def do_GET(self):
        if self.path.startswith('/_stat/'):
            self.send_response(server.HTTPStatus.OK)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(VideoFeed.get_stat()).encode())
            return
        if self.path.startswith('/_refresh_slides'):
            self._is_cam = False
            Thread(name="refreshSlides", target=refresh_slides, args=(1,), daemon=True).start()
            self.send_response(server.HTTPStatus.ACCEPTED)
            self.end_headers()
            self.close_connection = True
            return
        if not self.path.startswith('/cam/'):
            self._is_cam = False
            return super(MyResponder, self).do_GET()
        cam_no: str = self.path.split('/', 3)[2]
        if not cam_no.isdigit():
            self.send_error(server.HTTPStatus.NOT_FOUND, "File not found")
        cam_no = int(cam_no)
        self._is_cam = False
        self.send_response(server.HTTPStatus.OK)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0')
        self.send_header('Expires', 'Mon, 3 Jan 2000 12:34:56 GMT')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Connection', 'Keep-Alive')
        self.end_headers()
        self.close_connection = False
        VideoFeed.start_translation(self, cam_no, '?stream' not in self.path)
        logger.info(f"start serve video of camera #{cam_no} for {self.client_address}")
        # print(f"start serve video of camera #{cam_no} for {self.client_address}")
        return

    def do_HEAD(self):
        if not self.path.startswith('/cam/'):
            return super(MyResponder, self).do_HEAD()
        cam_no: str = self.path.split('/', 2)[2]
        if not cam_no.isdigit():
            self.send_error(server.HTTPStatus.NOT_FOUND, "File not found")
        self.send_response(server.HTTPStatus.OK)
        self.send_header("Content-type", "image/jpeg")
        self.end_headers()
        return

    def finish(self):
        if not self._is_cam:
            return super(MyResponder, self).finish()
        # prevent close file
        self.wfile.flush()


def send_frame(cam_no, img, frame_no):
    VideoFeed.send_frame(cam_no, img, frame_no)


webserver: server.HTTPServer = None
webserver_thread = None
watch_dog_thread = None


def start_webserver(port=8080, in_thread=True, daemon=True):
    global webserver, webserver_thread, watch_dog_thread
    import settings
    webserver = server.ThreadingHTTPServer(
        ('0.0.0.0', port),
        partial(MyResponder, directory=settings.SLIDES_PATH)
    )
    watch_dog_thread = Thread(name="WebWatchDog", target=watch_dog_loop, daemon=True, args=(VideoFeed.video_feeds,))
    watch_dog_thread.start()
    if in_thread:
        webserver_thread = Thread(name="webserver", target=webserver.serve_forever, daemon=daemon)
        webserver_thread.start()
    else:
        webserver.serve_forever()
