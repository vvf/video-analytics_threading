from pathlib import Path

import cv2
import datetime

import itertools

import os
import requests
import logging
from functools import partial

from redis import StrictRedis

import settings
from threading import Thread

logger = logging.getLogger(__name__)
redis = StrictRedis()

WAIT_IMAGE_MSG_DIVIDER = '***'


def add_wait_image_chat(cam_no, chat_id, message_id):
    logger.info(f'add_wait_image_chat: wait image from camera {cam_no} to {chat_id} replace message {message_id}')
    redis.sadd(f'wait_image_{cam_no}', f'{chat_id}{WAIT_IMAGE_MSG_DIVIDER}{message_id}')


def get_wait_image_chats(cam_no):
    while True:
        item = redis.spop(f'wait_image_{cam_no}')
        if not item:
            break
        result = tuple((item.decode().split(WAIT_IMAGE_MSG_DIVIDER) + [None])[:2])
        logger.info(f'get_wait_image_chats: Extracted image waiters: {result}')
        yield result


def has_wait_image_chats(cam_no):
    return redis.scard(f'wait_image_{cam_no}') > 0


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

    def message_to_subscribers(self, subscriber_type, text, **kwargs):
        dont_add_admins = kwargs.pop('dont_add_admins', None)
        subscribers = {m.decode() for m in redis.smembers(subscriber_type)}
        if not dont_add_admins:
            subscribers |= set(settings.BOT_ADMINS)
        for subscriber_id in subscribers:
            self.text(subscriber_id, text, **kwargs)

    def photo_to_subscribers(self, subscriber_type, photo, **kwargs):
        dont_add_admins = kwargs.pop('dont_add_admins', None)
        subscribers = {m.decode() for m in redis.smembers(subscriber_type)}
        if not dont_add_admins:
            subscribers |= set(settings.BOT_ADMINS)
        for subscriber_id in subscribers:
            self.sendPhoto(subscriber_id, photo, **kwargs)

    def sendPhoto(self, chat_id, photo, **kwargs):
        # here camelCase because it overrides APIs method
        bot_url = 'https://api.telegram.org/bot' + settings.TG_BOT_TOKEN
        files = {'photo': photo}
        kwargs['chat_id'] = chat_id
        resp = requests.post(bot_url + '/sendPhoto', data=kwargs, files=files)
        return resp.json().get('result')

    def sendVideo(self, chat_id, video, **kwargs):
        # here camelCase because it overrides APIs method
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
cam_noms = list(range(3))

_bot_running = True
bot_thread = None
# WAIT_MOTION_CHATS_KEY = "bot_wait_motion_chats"
WAIT_MOTION_CHATS_KEY = "motion_subscribers"
STATE_KEY = 'bot_chat_state'

AVAILABLE_SUBSCRIBES = {
    'known_cars': 'известная машина в зоне ворот',
    'motion': "любое движение перед уличной камерой",
    'monitor_person': "человек в зоне мониторигна",
    'monitor_car': "машина в зоне мониторигна",
    'monitor_any': "человек или машина в зоне мониторигна",
    'car_license_no': "появление машины с похожим номером (нужно указать номер, не менее 3х символов)",
    'cam1': "Движение перед камерой над входом",
    'cam2': "Движение перед камерой cam2"
}


def count_of_users_waiting_motion():
    return redis.scard(WAIT_MOTION_CHATS_KEY)


def is_user_waiting_motion(chat_id: str):
    return redis.sismember(WAIT_MOTION_CHATS_KEY, str(chat_id))


def stop_waiting_motion(chat_id: str):
    return redis.srem(WAIT_MOTION_CHATS_KEY, str(chat_id))


def start_waiting_motion(chat_id: str):
    return redis.sadd(WAIT_MOTION_CHATS_KEY, str(chat_id))


def add_bot_handler(bot_handler):
    bot_handlers.append(bot_handler)
    return bot_handler


def bot_loop():
    global _bot_running
    last_offset = None
    while _bot_running:
        try:
            updates = bot.getUpdates(offset=last_offset, limit=5, timeout=900)
        except Exception as error:
            logger.exception(error)
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
def bot_check_commands(update):
    if not 'message' in update:
        return
    message = update['message']
    chat_id = message['chat']['id']
    is_command = message.get('text', ' ').startswith('/')
    command_args = message.get('text', ' ').split(' ')
    if command_args and is_command:
        command = command_args.pop(0)
        logger.debug(f"got command: {command}")
    else:
        command = ''
    if command == '/get_image':
        if not command_args or command_args[0] == 'all':
            cam_no = cam_noms[:]
        else:
            cam_no = [int(c) for c in command_args if c.isdigit() and int(c) in cam_noms]
        msg = bot.answer(update, "Вам сейчас будет прислана фото с камеры")
        # msg = bot.answer(update, f"Вам сейчас будет прислана фото с камеры {cam_no}")
        for c in cam_no:
            add_wait_image_chat(c, chat_id, msg['message_id'])
        bot.sendChatAction(chat_id=chat_id, action='upload_photo')
        return True

    if command == '/wait_motion':
        if is_user_waiting_motion(chat_id):
            bot.answer(update, "Вы уже подписаны на движения")
            return
        start_waiting_motion(chat_id)
        bot.answer(update, "Вам будет прислано фото когда будет обнаружено движение")
        return True
    if command == '/stop_motion':
        if not is_user_waiting_motion(chat_id):
            bot.answer(update, "Вы и не подписаны на движения")
            return
        stop_waiting_motion(chat_id)
        bot.answer(update, "Вам больше не будут приходить фото при обнаружении движения")
        return True
    if command == '/get_video':
        bot.answer(update, "За какой день:", reply_markup={
            'inline_keyboard': [
                [inline_button(
                    '{:%d.%m.%Y}'.format(datetime.datetime.now() - datetime.timedelta(days=d)),
                    f'get_video/{d}'
                )] for d in range(5)
            ]
        })
        return True

    if command == '/sub':
        logger.debug(f"subscribe command, arguments: {command_args}")
        if not command_args:
            logger.debug(f"subscribe command, not arguments, response usage hint ")
            answer_subscribe_hint(update, command, 'Команда "Подписаться на уведомления".')
            return True
        subscribe_type = command_args[0]
        if subscribe_type == 'show':
            bot.answer(update, 'Вы подписаны на {}'.format(
                ', '.join(k.decode()[:-12].replace('_', '-')
                          for k in redis.keys('*_subscribers') if redis.sismember(k, chat_id))))
            return True
        if subscribe_type not in AVAILABLE_SUBSCRIBES:
            answer_subscribe_hint(update, command, 'Команда "Подписаться на уведомления"\nУкажите правильный аргумент.')
            return True
        if subscribe_type == 'car_license_no':
            # TODO: here add check argument to valud chars
            if len(command_args) < 2 or 4 > len(command_args[1]) > 9 or not command_args[1].isalnum():
                answer_subscribe_hint(update, command,
                                      'Команда "Подписаться на уведомления о машине с номерным знаком"'
                                      '\nУкажите правильный аргумент номер машины.')
                return True
            license_no = command_args[1]
            redis_key = f'car_license_no_{license_no.upper()}_subscribers'
        else:
            redis_key = f'{subscribe_type}_subscribers'
        redis.sadd(redis_key, chat_id)
        bot.answer(update, f'Вы подписались на уведомления "{AVAILABLE_SUBSCRIBES[subscribe_type].split("(", 1)[0]}"')
        return True

    if command == '/unsub':
        if not command_args:
            answer_subscribe_hint(update, command, 'Команда "Отписаться от уведомлений".')
            return True
        subscribe_type = command_args[0]
        if subscribe_type not in AVAILABLE_SUBSCRIBES:
            answer_subscribe_hint(update, command, 'Команда "Отписаться от уведомлений"\nУкажите правильный аргумент.')
            return True
        redis_key = f'{subscribe_type}_subscribers'
        redis.srem(redis_key, chat_id)
        bot.answer(update, f'Вы отписаись от уведомлений "{AVAILABLE_SUBSCRIBES[subscribe_type]}"')
        return True
    return False


def answer_subscribe_hint(update, command, txt):
    message = f"{txt}\nДля этой команды обязателен аргумент:\n" + \
              '\n'.join(f'  {k} - {v}' for k, v in AVAILABLE_SUBSCRIBES.items())
    bot.answer(
        update,
        message,
        parse_mode="HTML"
    )


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
    message_id = kwargs.pop('message_id', None)
    logger.info(f"Send file {kwargs['video']} to user {kwargs['chat_id']}")
    bot.sendVideo(**kwargs)
    logger.info(f"Done send file {kwargs['video']} to user {kwargs['chat_id']}")
    if message_id:
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
    if not has_wait_image_chats(cam_no):
        return
    rv, jpg = cv2.imencode('.jpeg', image)
    caption = f"Photo from camera #{cam_no}"
    if frame_no:
        caption += f" (frame {frame_no})"
    if rv:
        sended_count = 0
        for u, message_id in get_wait_image_chats(cam_no):
            bot.sendPhoto(chat_id=u, photo=jpg, caption=caption)
            if message_id:
                bot.deleteMessage(chat_id=u, message_id=message_id)
            sended_count += 1
        bot.message_to_admin(f"sended_count  of {cam_no} = {sended_count}")
    else:
        bot.message_to_admin("Error converting image from {cam_no}")


next_sent_motion_time = None


def send_motion_start(image, motion_id):
    global next_sent_motion_time
    if next_sent_motion_time and next_sent_motion_time > datetime.datetime.now():
        return 'time'
    rv, jpg = cv2.imencode('.jpeg', image)
    next_sent_motion_time = datetime.datetime.now() + datetime.timedelta(seconds=10)
    if rv:
        for u in redis.sscan_iter(WAIT_MOTION_CHATS_KEY):
            bot.sendPhoto(chat_id=u.decode(), photo=jpg, title=f"Motion id:{motion_id}")
    else:
        bot.message_to_admin("Error converting image")
    # bot.message_to_admin(f"len(bot_wait_image_chats)={len(bot_wait_image_chats)}")


def send_photo_to_admins(image, **kwargs):
    rv, jpg = cv2.imencode('.jpeg', image)
    if rv:
        img_ret = bot.send_image_to_amdin(jpg, **kwargs)
        # if kwargs.get('caption'):
        #     bot.message_to_admin(kwargs['caption'], disable_notification=True)
        return img_ret
    else:
        return bot.message_to_admin("Error converting image")


def send_monitoring_photo(image, is_person_or_car: bool, **kwargs):
    rv, jpg = cv2.imencode('.jpeg', image)
    if not rv:
        return bot.message_to_admin("Error converting image")
    bot.send_image_to_amdin(jpg, **kwargs)

    logger.debug(f"Send to monitor_any subscribers:{len(redis.smembers('monitor_any_subscribers'))}")
    for u in redis.sscan_iter('monitor_any_subscribers'):
        logger.debug(f"Send monitor_any to {u.decode()}")
        bot.sendPhoto(chat_id=u.decode(), photo=jpg, **kwargs)

    redis_key = 'monitor_{}_subscribers'.format(
        'person' if is_person_or_car else 'car'
    )
    for u in redis.sscan_iter(redis_key):
        logger.debug(f"Send {redis_key} to {u.decode()}")
        bot.sendPhoto(chat_id=u.decode(), photo=jpg, **kwargs)


def has_subscribers(event_name: str):
    redis_key = f'{event_name}_subscribers'
    return redis.scard(redis_key) > 0


def send_event_photo(image, event_name: str, **kwargs):
    redis_key = f'{event_name}_subscribers'
    # bot.send_image_to_amdin(jpg, **kwargs)
    jpg = None
    for u in redis.sscan_iter(redis_key):
        logger.debug(f"Send {redis_key} to {u.decode()}")
        if jpg is None:
            rv, jpg = cv2.imencode('.jpeg', image)
            if not rv:
                return bot.message_to_admin("Error converting image")

        bot.sendPhoto(chat_id=u.decode(), photo=jpg, **kwargs)


def send_event_video(event_name: str, video_file_path, **kwargs):
    redis_key = f'{event_name}_subscribers'
    for chat_id_bytes in redis.sscan_iter(redis_key):
        logger.debug(f"Send video to {redis_key} to {chat_id_bytes.decode()}")
        Thread(target=send_video_and_del_message,
               daemon=True,
               kwargs=dict(
                   chat_id=chat_id_bytes.decode(),
                   video=video_file_path,
                   width=1920,
                   height=1080
               )).start()
