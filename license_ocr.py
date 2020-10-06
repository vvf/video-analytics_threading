#!/usr/bin/env python3
import re

import cv2
import functools
import pytesseract
from datetime import datetime, timedelta
from redis import Redis
import os
from settings import ON_CAR_REACTIONS as reactions, ON_KNOWN_CAR_COMMAND

from modules.tgbot import bot

# https://medium.com/@jaafarbenabderrazak.info/ocr-with-tesseract-opencv-and-python-d2c4ec097866
# https://dropbox.tech/machine-learning/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning
# https://github.com/clovaai/CRAFT-pytorch

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
reactions_repeats = {}


print = functools.partial(print, flush=True)

def try_OCR_file(fname):
    img = cv2.imread(fname)
    if img is None:
        print(f'Can`t read {fname}')
        return ''
    h, w = img.shape[:2]
    if w < 50:
        print(f'\t{fname} too small')
        return ''
    text: str = pytesseract.image_to_string(img, config=tesseract_config)
    if not text:
        return ''
    text: str = text.upper()
    for src, dst in substitutions.items():
        if src not in char_whitelist:
            text = text.replace(src, dst)
    text = invalid_chars_re.sub('', text)
    print(f'{fname} : {text}')
    return text


known_car_timeout = datetime.now()
print(f"Wait for filenames. queue len at start -{redis.llen(queue)}\n\n")
while True:
    _, msg = redis.blpop(queue)
    filename = msg.decode()

    license_no_text = try_OCR_file(filename).upper()
    if not license_no_text or len(license_no_text) < 3:
        continue
    for car_no, message_text in reactions.items():
        timeout = reactions_timeouts.get(message_text)
        if car_no not in license_no_text:
            continue
        print(f'{car_no}\t{message_text}')
        if timeout and timeout > datetime.now():
            continue
        bot.message_to_subscribers('known_cars_subscribers', message_text)
        if not 18 > datetime.now().hour > 8:
            os.system(ON_KNOWN_CAR_COMMAND.format(message_text=message_text, car_no=car_no))
        known_car_timeout = datetime.now() + timedelta(minutes=10)
        reactions_repeats[message_text] = reactions_repeats.get(message_text, 0) + 1
        reactions_timeouts[message_text] = known_car_timeout + timedelta(
            minutes=reactions_repeats[message_text]*reactions_repeats[message_text]*5
        )
        if timeout and datetime.now().date() != timeout.date():
            reactions_repeats[message_text] = 1

    for subscribed_to_no in redis.keys(f'car_license_no_*_subscribers'):
        license_part = subscribed_to_no.decode()[29:]
        if license_part not in license_no_text:
            continue
        with open(filename, 'rb') as f:
            photo_content = f.read()
        bot.photo_to_subscribers(subscribed_to_no.decode(),photo_content, dont_add_admins=True)
