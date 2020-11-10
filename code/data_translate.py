import json
import random
import hashlib
from urllib import parse
import http.client

class BaiduTranslate:

    def __init__(self, fromLang, toLang):
        self.url = "/api/trans/vip/translate"
        # self.appid="" #申请的账号
        # self.secretKey = ''#账号密码
        # self.appid=""
        self.fromLang = fromLang
        self.toLang = toLang
        # self.salt = random.randint(32768, 65536)

    def BdTrans(self, text, try_times=2):
        appid, secretKey = random.choice(
            []) # 百度翻译秘钥，多几个避免被封了

        salt = random.randint(32768, 65536)

        # print(appid, secretKey)
        sign = appid + text + str(salt) + secretKey
        md = hashlib.md5()
        md.update(sign.encode(encoding='utf-8'))
        sign = md.hexdigest()
        myurl = self.url + \
                '?appid=' + appid + \
                '&q=' + parse.quote(text) + \
                '&from=' + self.fromLang + \
                '&to=' + self.toLang + \
                '&salt=' + str(salt) + \
                '&sign=' + sign
        try:
            httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
            httpClient.request('GET', myurl)
            response = httpClient.getresponse()
            html = response.read().decode('utf-8')
            html = json.loads(html)
            dst = html["trans_result"][0]["dst"]
            return dst

        except Exception as e:
            if try_times > 0:
                self.BdTrans(text, try_times-1)

            return None

if __name__=='__main__':
    BaiduTranslate_test = BaiduTranslate('zh', 'en')
    Results = BaiduTranslate_test.BdTrans("上古时代")#要翻译的词组

    print(Results)