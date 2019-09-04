from threading import Thread
from time import sleep

from modules.rtsp_reader import RTSPReaderThread
from modules.main import read_frame, LongReadFrameError
from modules import webserver
from settings import rtsp_url1, rtsp_url, rtsp_url2
cameras_urls = [
    rtsp_url,
    rtsp_url1,
    rtsp_url2
]
is_looping = True


def camera_loop(cam_url: str, cam_no: int):
    global is_looping
    rtsp_reader = RTSPReaderThread(rtsp_url=cam_url, daemon=True)
    rtsp_reader.start()
    try:
        frame_no = 0
        while is_looping:
            img = None
            try:
                img = read_frame(rtsp_reader, long_wait_behaviour=1, wait_timeout=30, cam_no=f'web-{cam_no}')
            except LongReadFrameError:
                rtsp_reader.open_video()
                print(f"reopen camera #{cam_no}")
            if img is not None:
                webserver.send_frame(cam_no, img, frame_no)
            frame_no += 1
        print(f"normal shutdown camera #{cam_no}")
    finally:
        is_looping = False
        rtsp_reader.stop()


if __name__ == '__main__':
    threads = []
    try:
        for cam_no, cam_url in enumerate(cameras_urls):
            t = Thread(name=f"webcam_{cam_no}", target=camera_loop, args=(cam_url, cam_no), daemon=True)
            t.start()
            threads.append(t)
        webserver.start_webserver(in_thread=False)
    finally:
        is_looping = False
        webserver.close_translations()
        print("Shutting down webserver")
        webserver.webserver.shutdown()
        print("Shutting down threads...")
        for t in threads:
            t.join(3)
        sleep(2)