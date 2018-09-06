import urllib
from bs4 import BeautifulSoup
import tqdm
import json
import os

class WebToonCrawler(object):
    """Download the entire webtoon images"""

    def __init__(self, ):

        self.DATA_DIR = 'webtoonsDB'
        self.WETOONS_JSON = os.path.join(self.DATA_DIR, 'webtoons.json')
        self.WEBTOONS = 'https://webtoons.com/en'
        self.dataset = {}

        if os.path.exists(self.DATA_DIR):
            print('FOLDER EXISTS')
        else:
            os.mkdir(self.DATA_DIR)
        if os.path.exists(self.WETOONS_JSON):
            print('FILE EXISTS')


    def gettoonlist(self):
        """get list of all webtoons in https://webtoons.com"""

        genre_page = self.WEBTOONS + "/genre"

        req = urllib.request.urlopen(genre_page)
        soup = BeautifulSoup(req, 'html.parser')

        card_items = soup.find_all('a', class_='card_item', href=True)
        webtoon_list = [line['href'] for line in card_items]
        webtoon_list = list(set(webtoon_list))

        return webtoon_list

    def getimagelist(self,webtoon_url):
        """get url of images of a webtoon
        Args:
            toon_url: url of a webtoon from gettoonlist
        returns:
            {'webtoon title':['url of images of a webtoon']}
        """

        webtoon_title = webtoon_url.split('/')[5]

        print('[INFO] Processing "{}"'.format(webtoon_title))

        img_list = {}
        epi_list = []


        """get page list of a webtoon"""
        req = urllib.request.Request(webtoon_url)
        req = urllib.request.urlopen(req)
        soup = BeautifulSoup(req, 'html.parser')


        tmp_page_list = soup.find_all('div', class_='paginate')
        tmp_page_list = [line.text.split('\n') for line in tmp_page_list][0]
        tmp_page_list = list(filter(lambda x : x != '', tmp_page_list))

        while tmp_page_list[-1] == 'Next Page':
            tmp_page_list.pop()
            next_page = str(int(tmp_page_list[-1]) + 1)

            req = urllib.request.Request(webtoon_url + '&page=' + next_page)
            req = urllib.request.urlopen(req)
            soup = BeautifulSoup(req, 'html.parser')

            tmp_tmp_page_list = soup.find_all('div', class_='paginate')
            tmp_tmp_page_list = [line.text.split('\n') for line in tmp_tmp_page_list][0]
            tmp_tmp_page_list = list(filter(lambda x : x != '', tmp_tmp_page_list))

            tmp_page_list += tmp_tmp_page_list
        tmp_page_list = list(filter(lambda x : x != 'Previous Episode', tmp_page_list))

        print('[INFO] page list: {}'.format(tmp_page_list))


        """get list of episode urls in a page"""
        for page in tmp_page_list:
            webtoon_page =  webtoon_url + '&page=' + page
            req =  urllib.request.urlopen(webtoon_page)
            soup = BeautifulSoup(req, 'html.parser')

            tmp_epi_list = [a['href'] for a in soup.select('.detail_lst ul li a')]
            epi_list += tmp_epi_list

        print('[INFO] crawling "{}"`s total images...'.format(webtoon_title))


        """get list of images of a webtoon"""
        for epi in tqdm.tqdm(epi_list):
            epi_no = epi.split('=')[-1]

            req = urllib.request.Request(epi)
            req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0")
            r = urllib.request.urlopen(req)
            soup = BeautifulSoup(r, 'html.parser')

            tmp_img_list = soup.find_all('img', class_='_images')
            tmp_img_list = [img['data-url'] for img in tmp_img_list]
            img_list.update({epi_no:tmp_img_list})


        """put webtoon images in datasets"""
        self.dataset.update({webtoon_title:img_list})


    def savetojson(self, file):
        with open(self.WETOONS_JSON, 'w') as f:
            json.dump(file, f, ensure_ascii=False)
            f.close()


def run():
    crawler = WebToonCrawler()
    webtoon_list = crawler.gettoonlist()
    for url in tqdm.tqdm(webtoon_list):
        crawler.getimagelist(url)
    print(crawler.dataset)
    crawler.savetojson(crawler.dataset)

if __name__ == '__main__':
    run()