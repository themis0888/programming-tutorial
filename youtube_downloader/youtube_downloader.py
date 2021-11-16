import sys
import os
from multiprocessing.pool import ThreadPool

import youtube_dl, ffmpeg

ydl_opts = {
    'format': '22/18',
    'quiet': True,
    'ignoreerrors': True,
    'no_warnings': True,
}
yt_id = 'QoQF8N5ZsQA'
yt_base_url = 'https://www.youtube.com/watch?v='
yt_url = yt_base_url+yt_id

ydl_opts = {
    'format': '22/18',
    'quiet': True,
    'ignoreerrors': True,
    'no_warnings': True,
}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    download_url = ydl.extract_info(url=yt_url, download=False)['url']


input_file = ffmpeg.input(download_url, ss=240.006433, to=244.961389)
output_file = input_file.output('output.mp4', format='mp4', r=25, vcodec='libx264',
            crf=18, preset='veryfast', pix_fmt='yuv420p', acodec='aac', audio_bitrate=128000,
            strict='experimental')
    
output_file.global_args('-y').global_args('-loglevel', 'error').run()