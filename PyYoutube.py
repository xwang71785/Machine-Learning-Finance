# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 09:59:43 2017
Youtube Download
@author: wangx3
"""

import ssl
import requests
import re
import json
import urllib.request as req
import urllib.parse as par
import shutil

context = ssl._create_unverified_context()

res = req.urlopen('https://www.youtube.com/watch?v=NOkp3rEGeyE', context=context)
data = res.read()

print('Status:', res.status, res.reason)
for k, v in res.getheaders():
    print('%s: %s' % (k, v))
# print('Data:', data.decode('utf-8'))
html = data.decode('utf-8')

m = re.search('"args":({.*?}),', html)

jd = json.loads(m.group(1))
print(jd["url_encoded_fmt_stream_mp"])
'''

a = par.parse_qs(jd["url_encoded_fmt_stream_mp"])
print(a['url'][0])

res2 = requests.get(a['url'][0], stream=True)
f = open('a.mp4', 'wb')
shutil.copyfileobj(res2.raw, f)
f.close()


from __future__ import unicode_literals
import youtube_dl
import urllib
import shutil
ydl_opts = {}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
ydl.download(['https://www.youtube.com/watch?v=n06H7OcPd-g'])
'''
