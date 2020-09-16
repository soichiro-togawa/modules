import logging
import datetime
from pytz import timezone
import os,sys

# LOG_DIR = ""

def setup_logger(LOG_DIR, out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    print("ログセットアップ")
    LOGGER = logging.getLogger()

    #タイムゾーンを東京にコンバート
    def customTime(*args):
      return datetime.datetime.now(timezone('Asia/Tokyo')).timetuple()
    FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    FORMATTER.converter = customTime

    # FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")

    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    return LOGGER


#########使用方法##########
# #loggerのセットアップ
# import datetime
# from pytz import timezone
# from make_log import setup_logger
# file_name = "model_log"
# LOG_DIR = "/"
# NOW = datetime.datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d_%H:%M:%S')
# logger = setup_logger(LOG_DIR,os.path.join(LOG_DIR,'{0}_{1}.log'.format(file_name,NOW)))
# logger.info("log:{0}".format("comment"))