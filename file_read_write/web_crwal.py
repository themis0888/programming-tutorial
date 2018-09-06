import json
import urllib.request
import os
import tqdm

class ImageDownLoader(object):
    """webtoons image downloader"""

    def __init__(self, jsonfile):
        """set data directories of dataset
        Args:
            jsonfile : jsonfile generated from WebToonCrawler
        """

        self.DATA_DIR = '/home/siit/navi/data/webtoonsDB'
        self.IMAGE_DATA = os.path.join(self.DATA_DIR, 'imageset')
        self.JSON_DIR = os.path.join(self.DATA_DIR, jsonfile)
        self.JSON = json.load(open(self.JSON_DIR))


    def download(self):
        """download webtoon image"""

        choosed_webtoon = ['free-draw']

        if not os.path.exists(self.IMAGE_DATA):
            os.mkdir(self.IMAGE_DATA)

        for webtoon, img in self.JSON.items():
            WEBTOON_PATH = os.path.join(self.IMAGE_DATA, webtoon)
            if os.path.exists(WEBTOON_PATH):
                print('[INFO] FOLDER EXISTS')
                #continue
            else:
                os.mkdir(WEBTOON_PATH)
                
            ##################################################################################
            """control number of episodes to download if webtoon is not in choosed_webtoon"""#
            ##################################################################################
            if webtoon not in choosed_webtoon:
                for k in range(6,len(img)+1):
                    try:
                        del img[str(k)]
                    except:
                        print('{} not exists'.format(k))

            """start downloding"""
            for epi, img_url_list in tqdm.tqdm(img.items()):
                EPISODE_PATH = os.path.join(WEBTOON_PATH, epi)

                if os.path.exists(EPISODE_PATH):
                    print("[INFO] episode {} exists".format(epi))
                else:
                    os.mkdir(EPISODE_PATH)
                    try:
                        for i, img_url in enumerate(img_url_list):

                            req = urllib.request.Request(img_url)
                            req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0")
                            req.add_header('Referer', 'https://comic.naver.com')
                            with open(os.path.join(EPISODE_PATH, "{}_{}.png".format(webtoon, i)), 'wb') as f:
                                try:
                                    f.write(urllib.request.urlopen(req).read())
                                except:
                                    print("Failed {}_{}.png".format(webtoon, i))
                                    with open(os.path.join(self.DATA_DIR, 'failed_list.txt', 'w')) as f:
                                        f.write("{}_{}.png".format(webtoon, i))
                    except:
                        print("{} {} exists".format(webtoon, epi))


def run():
    downloader = ImageDownLoader('webtoons.json')
    downloader.download()

if __name__ == '__main__':
    run()