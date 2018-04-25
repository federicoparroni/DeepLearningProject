import time
import datetime
import urllib3

import telegram


telegram_default_token = '591311395:AAEfSH464BdXSDezWGMZwdiLxLg2_aLlGDE'
telegram_default_chat_id = -1001223624517
telegram_default_timeout = 100


def current_datetime():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def connection_available():
    try:
        urllib3.urlopen('http://216.58.192.142', timeout=1)
        return True
    except urllib3.URLError as _:
        return False


# Send a message through telegram
def telegram_send_msg(message, timeout=telegram_default_timeout, token=telegram_default_token, chat_id=telegram_default_chat_id):
    try:
        bot = telegram.Bot(token=token)
        message = "{} - {}".format(current_datetime(), message)
        bot.send_message(chat_id=chat_id, text=message, timeout=timeout)
    except Exception as err:
        print(err)
