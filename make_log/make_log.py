import logging
import datetime
from pytz import timezone
import os,sys

# LOG_DIR = ""

def setup_logger(LOG_DIR, FILE_NAME=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    # print("ログセットアップ、ログファイルの生成")
    LOGGER = logging.getLogger()

    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    NOW = datetime.datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d_%H:%M:%S')
    out_file = LOG_DIR + "{0}_{1}.log".format(NOW,FILE_NAME)

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



    return LOGGER


#########使用方法##########
# #loggerのセットアップ,一つのlogファイルに次々書き込まれていく方式
# from make_log import setup_logger
# #DIRがなければ自動作成
# LOG_DIR = "/content/log/"
# FILE_NAME = "training_log"
# logger = setup_logger(LOG_DIR, FILE_NAME)
# #生成したログファイルへの書き込み
# COMMENT1 = "comment1"
# COMMENT2 = "comment2"
# logger.info("log:{}".format(COMMENT1))
# logger.info("log:{}".format(COMMENT2))