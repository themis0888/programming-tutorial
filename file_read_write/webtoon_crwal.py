'''
                                              NaverWebtoonCrawler
                                                                                                            - Version: 0.0.2
Author: Minu Jeong
contact: minu.hanwool@gmail.com
Dependencies
    mechanize
    BeautifulSoup
'''

#####################   import modules   #####################
##  external modules  ##
# mechanize / cookielib modules for fake browsing
import mechanize, cookielib

# BeautifulSoup module for parsing HTML
from bs4 import BeautifulSoup

##  internal modules  ##
# time module for log timestamp
import time

# os, cStringIO modules for utility in control files
import os, cStringIO


#####################   constants   #####################
def WEB_TOON_ID() :
    # get webtoon id from naver webtoon url(?titleId=WEB_TOON_ID).
    # return 119874 # denma
    return 597447 # living in the fantasy world

def IS_VERBOSE() :
    # log if wanted
    return True

def SAVE_LOG() :
    # save log as a file ("log.txt")
    return True

#####################   class definitions   #####################
class NaverWebtoonCrawler :
# class properties
    @property # getter
    def browser(self) :
        # initialize __browser once
        if self.__browser__ == None:
            # if __browser__ never initialized,
            self.__browser__ = self.initializeBrowser()
            return self.__browser__
        else :
            # else just return __browser__
            return self.__browser__

    def initializeBrowser(self) :
        Log("new mechanize browser initialized")
        # Browser
        self.__browser__ = mechanize.Browser()

        # set cookie Jar
        self.__browser__.set_cookiejar(cookielib.LWPCookieJar())

        # Browser options
        self.__browser__.set_handle_robots(False) # ignore robots.txt restrictions
        self.__browser__.addheaders = [('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.117 Safari/537.36')] # ignore non-human restrictions
        self.__browser__.addheaders = [('Referer', 'http://comic.naver.com')] # ignore naver restriction
        return self.__browser__

    # methods
    def __init__(self) :
        Log("Crawler initialized.")

        # setup properties
        self.__browser__ = None
        return

    def __str__(self) :
        return "<Naver Webtoon Crawler>"

    # get data
    def getWebtoon(self, flip = 1) :
        Log("request chapter type: %s" % type(flip) )

        if type(flip) is type(0) :
            # in case chapter request is integer.
            self.saveToFile(flip)

        elif type(flip) is type([]) or type(flip) is type(()) :
            # in case chapter request is list or tuple.
            for chapterIndex in flip :
                self.saveToFile(chapterIndex)

        else :
            # leave a log: requested type is not supported.
            Log("[ERROR] request chapter type: %s is not supported. sorry." % type(flip) )

    def saveToFile(self, flip) :
        Log ("Request chapter: %s" % flip )
        # open browser, get BeautifulSoup
        bs = BeautifulSoup (  self.browser.open( "http://comic.naver.com/webtoon/detail.nhn?titleId=%s&no=%s" % (WEB_TOON_ID(), flip) ).read()  )

        # get title name using facebook open-graph meta-data
        flipName = bs.find("meta", {"property":"og:title"}).get("content").encode("utf-8")

        # create directory
        dirs = str(flip) + "_" + flipName # name of dir
        if not os.path.exists(dirs) :
            os.mkdir(dirs) # check exists
            Log("Directory not exists. creating %s folder." % dirs)


        # by using BeautifulSoup, grep img tags
        images = bs.find("div", {"class":"wt_viewer"}).findAll("img")
        pageNumber = 0 # page number for multiple files
        for image in images :
            src = image.get( "src" ) # grep src

            # filtering conditions
            if src == "" :
                continue # condition: must contain src
            if src.find(".jpg") == -1 :
                continue # condition: must be .jpg file

            pageNumber += 1 # start from 1

            data = self.browser.open(  src  ).read() # get data from browser

            Log ("attemp: save to file ")
            # write to file
            file = open("%s/page_%d.jpg" % (dirs, pageNumber), "w")
            file.write(data)
            file.close()

            Log(  "image saved file at %s/page_%d.jpg" % (dirs, pageNumber)  )

            # browser navigate to back
            # self.browser.back()
        return len(images) == 0

#####################   Log function   #####################
def Log(message) :
    if IS_VERBOSE() :
        # print a log if IS_VERBOSE is True
        print "[Log] Crawler Log: " + message + "  -- timestamp: @", int (  time.time()  )

    if SAVE_LOG() :
        # save a log if SAVE_LOG is True
        __logFile__ = open("crawler_log.txt", "ab+")
        __logFile__.write( "[Log] " + message + "   -- timestamp: @" + str(  int ( time.time() ) ) + "\n")
        __logFile__.close() # close file: leave a log to the file imediately.
    return


#####################   starting point   #####################
if __name__ == "__main__" :
    # initialize global variables
    Log("     ****** Crawler Executed ******    ")

    # initialize class
    crawl = NaverWebtoonCrawler()

    # get webtoon chapter: UNIT = chapter number
    # support list as parameter
    Log("Crawler.py executed.")
    crawl.getWebtoon(  list(range(0, 68))  ) # test

    # supporti tuple as parameter
    # denmaCrawler.getWebtoon(  (1, 2, 3, 4)  )

    '''
    # for multiple chapters in range
    for index in range(1, 2) :
        denmaCrawler.getWebtoon( index )
    '''