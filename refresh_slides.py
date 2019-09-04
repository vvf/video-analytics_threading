'''
Scan directory and send result to redis (and so to web page thought websocket)
'''
import json
from pathlib import Path
from redis import Redis
import cv2
import settings
import os
WIDTH = 1024
HEIGHT = 716

DESTPATH = settings.SLIDES_PATH

def resize(f):
    img = cv2.imread(f.as_posix())
    h, w = img.shape[:2]
    k = 1
    if w > h:
        k = WIDTH / w
    else:
        k = HEIGHT / h
    h = int(h * k)
    w = int(w * k)
    img = cv2.resize(img, (w, h))
    cv2.imwrite(os.path.join(DESTPATH, f.name.lower()), img)


def prepare_imgs(src_folder):
    p = Path(src_folder)
    for f in p.glob('**/*'):
        if not f.name.lower().endswith('.jpg') and not f.name.lower().endswith('.jpeg'):
            continue
        if '.AppleDouble' in str(f):
            continue
        if os.path.exists(os.path.join(DESTPATH, f.name.lower())):
            continue
        print(str(f))
        resize(f)


if __name__ == '__main__':
    import sys
    prepare_imgs(sys.argv[1] or './')
    p = Path(settings.SLIDES_PATH)
    redis = Redis()
    imgs = [f.as_posix().replace(settings.SLIDES_PATH, '') for f in p.glob('*') if
            f.is_file() and not f.name.startswith('.')]
    print(f"Sending {len(imgs)} slides")
    redis.publish('/ws/control', json.dumps({'action': 'refreshSlides', 'slides': imgs}))
