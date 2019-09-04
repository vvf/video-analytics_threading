#!/usr/bin/env python3
import re

import cv2
import pytesseract
from datetime import datetime, timedelta
from redis import Redis
import os
from settings import ON_CAR_REACTIONS as reactions, ON_KNOWN_CAR_COMMAND

from modules.tgbot import bot

char_whitelist = '1234567890ABEKMHOPCTYX'

invalid_chars_re = re.compile('[^' + char_whitelist + ']')
queue = os.environ.get('LICENSE_QUEUE_NAME', 'licenses')

substitutions = dict(['0O', 'S5', 'G6', 'B8', 'WM', 'Z2', 'I1', 'RP', 'L1'])
substitutions.update({
    v: k
    for k, v in substitutions.items()
    if k in char_whitelist
})

tesseract_config = "-l eng -oem 1 -psm 8 -c language_model_penalty_non_dict_word=0"
redis = Redis()
reactions_timeouts = {}


def try_OCR_file(fname):
    img = cv2.imread(fname)
    if img is None:
        return
    h, w = img.shape[:2]
    if w < 50:
        print(f'\t{fname} too small')
        return None
    text: str = pytesseract.image_to_string(img, config=tesseract_config)
    if not text:
        return None
    text: str = text.upper()
    for src, dst in substitutions.items():
        if src not in char_whitelist:
            text = text.replace(src, dst)
    text = invalid_chars_re.sub('', text)
    print(f'{fname} : {text}')
    return text


known_car_timeout = datetime.now()
print("Wait for filenames\n\n")
while True:
    _, msg = redis.blpop(queue)
    if known_car_timeout > datetime.now():
        print(f'skip {msg.decode("utf8")}')
        continue
    license_no_text = try_OCR_file(msg.decode())
    if not license_no_text:
        continue
    for car_no, message_text in reactions.items():
        timeout = reactions_timeouts.get(car_no)
        if car_no not in license_no_text:
            continue
        print(f'{car_no}\t{message_text}')
        if timeout and timeout > datetime.now():
            continue
        bot.message_to_admin(message_text)
        os.system(ON_KNOWN_CAR_COMMAND.format(message_text=message_text, car_no=car_no))
        known_car_timeout = datetime.now() + timedelta(minutes=1)
        reactions_timeouts[car_no] = known_car_timeout + timedelta(minutes=1)
