CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tv"]

COLORS = [
    [106, 207, 17],  # background
    [151, 148, 78],  # aeroplane
    [162, 26, 30],  # bicycle
    [252, 12, 167],  # bird
    [240, 250, 92],  # boat
    [206, 235, 79],  # bottle
    [185, 251, 162],  # bus
    [233, 47, 79],  # car
    [22, 136, 170],  # cat
    [35, 197, 228],  # chair
    [175, 94, 210],  # cow
    [192, 73, 6],  # diningtable
    [188, 62, 250],  # dog
    [20, 242, 45],  # horse
    [147, 199, 174],  # motorbike
    [204, 52, 38],  # person
    [95, 8, 65],  # pottedplant
    [190, 204, 68],  # sheep
    [202, 68, 244],  # sofa
    [149, 178, 19],  # train
    [233, 150, 177],  # tv
]

CAR_IDXES = {CLASSES.index('car'), CLASSES.index('aeroplane'), CLASSES.index('boat'), CLASSES.index('train'),
             CLASSES.index('sofa'), CLASSES.index('bus')}

PERSON_IDXES = {
    CLASSES.index('person'),
    CLASSES.index('motorbike'),
    CLASSES.index('horse'),
    CLASSES.index('bottle'),
    CLASSES.index('bird'),
    CLASSES.index('cow')
}
EXACTLY_PERSON_INDEX = CLASSES.index('person')
NOT_DRAW_CLASSES = {
    CLASSES.index('chair'),
    CLASSES.index('tv'),
    CLASSES.index('sofa'),
    CLASSES.index('bottle'),
    CLASSES.index('diningtable')
}

CAR_MIN_SQUARE = 90000  # square at least 300x300
PERSON_MIN_HEIGHT = 180

PRE_MOTION_FRAMES = 35  # 20 frames is pre motion and 5 frames when motion is detected
POST_MOTION_FRAMES = 35
FRAMES_OF_REAL_MOTION = 5

BLUR_PARAM = (11, 11)

SMALL_IMG_DIM = (480, 270)  # (576,324)
SMALL_IMG_K = SMALL_IMG_DIM[0] / 1920

ALARM_ZONE = ()

TG_BOT_TOKEN = '<here token of TG bot>'
BOT_ADMINS = ('<chat_id of admin>',)

SLIDES_PATH = '/home/vvf/share/slides'

ON_CAR_REACTIONS = {
    'A334PE': "Алеша приехал!",
    'A567PC': "Вася приехал!",
    'M737P0': "Папа приехал!",
    'M737PO': "Папа приехал!",
}
rtsp_url1 = 'rtsp://192.168.77.16:554/user=user&password=password&channel=1&stream=0.sdp?real_stream'
rtsp_url = 'rtsp://192.168.77.15:554/user=user&password=password&channel=1&stream=0.sdp?real_stream'

