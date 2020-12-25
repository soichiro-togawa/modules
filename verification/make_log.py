import logging
import datetime
from pytz import timezone
import os,sys

LOG_DIR = "./verification/_output_dir/"

#出力と、ファイル出力の二つのレベルにおいて設定(file_level=logging.INFO)
def setup_logger(LOG_DIR=LOG_DIR, FILE_NAME=None, stderr=True, stderr_level=logging.INFO, file_level=logging.INFO):
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

# time.timeを返すのが基本で、pytorchとgpu使ってるときは同期を含む
def time_synchronized():
    # pytorch-accurate time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

# from make_log.make_log import setup_logger
#FILE_NAME="yolov5s_evaltime_input848_batch1_fp16"
# logger = setup_logger(FILE_NAME)
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