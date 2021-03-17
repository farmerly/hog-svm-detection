import time


def get_local_time():
    """
    :return: 格式化后的时间字符串
    """
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
