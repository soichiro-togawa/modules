#loggerのセットアップ,一つのlogファイルに次々書き込まれていく方式
from make_log import setup_logger
LOG_DIR = "/content/log/"
LOG_NAME = "training_log"
a = "fruits"
def app(a):
    logger = setup_logger(LOG_DIR, LOG_NAME)
    print(a,"apple")
    COMMENT1 = a
    logger.info("log:{}".format(COMMENT1))

def run():
    #loggerのセットアップ,一つのlogファイルに次々書き込まれていく方式


    #生成したログファイルへの書き込み
    app(a)


    return a

