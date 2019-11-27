from pathlib import Path

import cv2
import datetime

import itertools

import os
import requests
import logging
from functools import partial

from redis import Redis

import settings
from threading import Thread

logger = logging.getLogger(__name__)
redis = Redis()


def call_api(method, **data):
    bot_url = 'https://api.telegram.org/bot' + settings.TG_BOT_TOKEN
    resp = requests.post(bot_url + '/' + method, json=data)
    return resp.json().get('result')


class BotApi:
    def __getattr__(self, method_name):
        method = partial(call_api, method_name)
        setattr(self, method_name, method)
        return method

    def answer(self, update, text, parse_mode='Markdown', **kwargs):
        return self.sendMessage(chat_id=update['message']['chat']['id'],
                                reply_to_message_id=update['message']['message_id'],
                                parse_mode=parse_mode,
                                text=text,
                                **kwargs)

    def text(self, chat_id, text, parse_mode='Markdown', **kwargs):
        if isinstance(chat_id, dict):
            chat_id = chat_id['message']['chat']['id']
        return self.sendMessage(chat_id=chat_id,
                                parse_mode=parse_mode,
                                text=text,
                                **kwargs)

    def message_to_admin(self, text, **kwargs):
        for admin_id in settings.BOT_ADMINS:
            self.text(admin_id, text, **kwargs)

    def sendPhoto(self, chat_id, photo, **kwargs):
        bot_url = 'https://api.telegram.org/bot' + settings.TG_BOT_TOKEN
        files = {'photo': photo}
        kwargs['chat_id'] = chat_id
        resp = requests.post(bot_url + '/sendPhoto', data=kwargs, files=files)
        return resp.json().get('result')

    def sendVideo(self, chat_id, video, **kwargs):
        bot_url = 'https://api.telegram.org/bot' + settings.TG_BOT_TOKEN
        bot.sendChatAction(chat_id=chat_id, action='upload_video')
        with open(video, 'rb') as f:
            files = {'video': f}
            kwargs['chat_id'] = chat_id
            resp = requests.post(bot_url + '/sendVideo', data=kwargs, files=files)
        return resp.json().get('result')

    def send_image_to_amdin(self, image, **kwargs):
        for admin_id in settings.BOT_ADMINS:
            last_result = self.sendPhoto(
                chat_id=admin_id,
                photo=image,
                **kwargs
            )
        return last_result


bot = BotApi()

bot_handlers = []
bot_wait_image_chats = []
_bot_running = True
bot_thread = None
WAIT_MOTION_CHATS_KEY = "bot_wait_motion_chats"
STATE_KEY = 'bot_chat_state'


def count_of_users_waiting_motion():
    return redis.scard(WAIT_MOTION_CHATS_KEY)


def is_user_waiting_motion(chat_id: str):
    return redis.sismember(WAIT_MOTION_CHATS_KEY, chat_id)


def stop_waiting_motion(chat_id: str):
    return redis.srem(WAIT_MOTION_CHATS_KEY, chat_id)


def start_waiting_motion(chat_id: str):
    return redis.sadd(WAIT_MOTION_CHATS_KEY, chat_id)


def add_bot_handler(bot_handler):
    bot_handlers.append(bot_handler)
    return bot_handler


def bot_loop():
    global _bot_running
    last_offset = None
    while _bot_running:
        try:
            updates = bot.getUpdates(offset=last_offset, limit=5, timeout=900)
        except Exception:
            updates = None
            pass
        if not updates:
            continue
        for update in updates:
            try:
                for handler in bot_handlers:
                    handler(update)
            except Exception as error:
                logger.exception(error)
            last_offset = update['update_id'] + 1


def stop_bot():
    global _bot_running
    _bot_running = False
    if bot_thread:
        bot_thread.join()


def start_bot():
    global bot_thread
    bot_thread = Thread(target=bot_loop, name='TgBot', daemon=True)
    bot_thread.start()


@add_bot_handler
def bot_get_image_cmd(update):
    if not 'message' in update:
        return
    message = update['message']
    chat_id = message['chat']['id']
    if message.get('text', ' ').split(' ')[0] == '/get_image':
        if chat_id in bot_wait_image_chats:
            return True
        msg = bot.answer(update, "Вам сейчас будет прислана фото с камеры")
        bot_wait_image_chats.append((chat_id, msg['message_id']))
        bot.sendChatAction(chat_id=chat_id, action='upload_photo')
        return True

    if message.get('text', ' ').split(' ')[0] == '/wait_motion':
        if is_user_waiting_motion(chat_id):
            bot.answer(update, "Вы уже подписаны на движения")
            return
        start_waiting_motion(chat_id)
        bot.answer(update, "Вам будет прислано фото когда будет обнаружено движение")
        return True
    if message.get('text', ' ').split(' ')[0] == '/stop_motion':
        if is_user_waiting_motion(chat_id):
            bot.answer(update, "Вы и не подписаны на движения")
            return
        stop_waiting_motion(chat_id)
        bot.answer(update, "Вам больше не будут приходить фото при обнаружении движения")
        return True
    if message.get('text', ' ').split(' ')[0] == '/get_video':
        bot.answer(update, "За какой день:", reply_markup={
            'inline_keyboard': [
                [inline_button(
                    '{:%d.%m.%Y}'.format(datetime.datetime.now() - datetime.timedelta(days=d)),
                    f'get_video/{d}'
                )] for d in range(5)
            ]
        })


@add_bot_handler
def bot_get_callbacks(update):
    if not 'callback_query' in update:
        return
    cbq = update['callback_query']
    message = cbq['message']
    chat_id = message['chat']['id']
    callback_data = cbq.get('data', '')
    if callback_data.startswith('get_video/'):
        get_video_params = callback_data.split('/')
        if len(get_video_params) == 2:
            bot.editMessageText(
                chat_id=chat_id,
                message_id=message['message_id'],
                text="За какой час:",
                reply_markup={
                    'inline_keyboard': get_hours_keyboard_of_day(get_video_params[1])
                }
            )
        elif len(get_video_params) == 3:
            bot.editMessageText(
                chat_id=chat_id,
                message_id=message['message_id'],
                text="Выберите какой файл прислать (время записи):",
                reply_markup={
                    'inline_keyboard': get_videos_keyboard_of_day(*get_video_params[1:])
                }
            )
        elif len(get_video_params) == 4:
            bot.editMessageText(
                chat_id=chat_id,
                message_id=message['message_id'],
                text="Видео загружается...",
                reply_markup={}
            )
            day, hour, fname = get_video_params[1:]
            now_ts = datetime.datetime.now() - datetime.timedelta(days=int(day))
            p = os.path.join(
                now_ts.strftime('%Y'),
                now_ts.strftime('%m'),
                now_ts.strftime('%d'),
                hour,
                fname
            )
            Thread(target=send_video_and_del_message,
                   daemon=True,
                   kwargs=dict(
                       chat_id=chat_id,
                       video=p,
                       width=1920,
                       height=1080,
                       message_id=message['message_id']
                   )).start()
    bot.answerCallbackQuery(callback_query_id=cbq['id'])


def send_video_and_del_message(**kwargs):
    message_id = kwargs.pop('message_id')
    logger.info(f"Send file {kwargs['video']} to user {kwargs['chat_id']}")
    bot.sendVideo(**kwargs)
    logger.info(f"Done send file {kwargs['video']} to user {kwargs['chat_id']}")
    bot.deleteMessage(
        chat_id=kwargs['chat_id'],
        message_id=message_id
    )


def get_hours_keyboard_of_day(day):
    now_ts = datetime.datetime.now() - datetime.timedelta(days=int(day))
    p = os.path.join(
        now_ts.strftime('%Y'),
        now_ts.strftime('%m'),
        now_ts.strftime('%d'),
    )
    path = Path(p)
    itertools.groupby(range(10), key=lambda x: x // 4)
    prefix = f'get_video/{day}/'
    return [[inline_button(hdir.name[1:], prefix + hdir.name) for npp, hdir in grp] for grp_no, grp in
            itertools.groupby(enumerate(sorted(path.glob('h*'))), key=lambda x: x[0] // 4)
            ]


def get_videos_keyboard_of_day(day, hour):
    now_ts = datetime.datetime.now() - datetime.timedelta(days=int(day))
    p = os.path.join(
        now_ts.strftime('%Y'),
        now_ts.strftime('%m'),
        now_ts.strftime('%d'),
        hour
    )
    path = Path(p)
    itertools.groupby(range(10), key=lambda x: x // 4)
    prefix = f'get_video/{day}/{hour}/'
    return [[inline_button(time_from_motion_filename(hdir.name), prefix + hdir.name) for npp, hdir in grp] for
            grp_no, grp in
            itertools.groupby(enumerate(sorted(path.glob('motion_*'))), key=lambda x: x[0] // 4)
            ]


def time_from_motion_filename(motion_fname):
    return ':'.join(motion_fname[16 + i * 2:][:2] for i in range(3))


def inline_button(text, callback_data, url=None):
    result = {
        'text': text,
        'callback_data': callback_data
    }
    if url:
        result['url'] = url
    return result


def send_photo_if_need(cam_no, image, frame_no=None):
    if not bot_wait_image_chats:
        return
    send_to = [bot_wait_image_chats.pop() for i in range(len(bot_wait_image_chats))]
    rv, jpg = cv2.imencode('.jpeg', image)
    caption = f"Photo from camera #{cam_no}"
    if frame_no:
        caption += f" (frame {frame_no})"
    if rv:
        for u, message_id in send_to:
            bot.sendPhoto(chat_id=u, photo=jpg, caption=caption)
            bot.deleteMessage(chat_id=u, message_id=message_id)
    else:
        bot.message_to_admin("Error converting image")
    bot.message_to_admin(f"len(bot_wait_image_chats)={len(bot_wait_image_chats)}")


next_sent_motion_time = None


def send_motion_start(image):
    global next_sent_motion_time
    if next_sent_motion_time and next_sent_motion_time > datetime.datetime.now():
        return 'time'
    rv, jpg = cv2.imencode('.jpeg', image)
    next_sent_motion_time = datetime.datetime.now() + datetime.timedelta(seconds=10)
    if rv:
        for u in redis.sscan_iter(WAIT_MOTION_CHATS_KEY):
            bot.sendPhoto(chat_id=u.decode(), photo=jpg)
    else:
        bot.message_to_admin("Error converting image")
    bot.message_to_admin(f"len(bot_wait_image_chats)={len(bot_wait_image_chats)}")


def send_photo_to_admins(image, **kwargs):
    rv, jpg = cv2.imencode('.jpeg', image)
    if rv:
        img_ret = bot.send_image_to_amdin(jpg, **kwargs)
        # if kwargs.get('caption'):
        #     bot.message_to_admin(kwargs['caption'], disable_notification=True)
        return img_ret
    else:
        return bot.message_to_admin("Error converting image")
