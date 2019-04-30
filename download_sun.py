#!/usr/bin/env python3
# coding: utf-8

from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

import ipdb
import urllib.request
import os

import eventlet

def fetch(url, filename):
    print("downloading " + str(filename))
    if not os.path.exists(filename):
        try:
            urllib.request.urlretrieve(url, filename)
        finally:
            print("done " + str(filename))
    else:
        print("already downloaded")
    return

def download_day(url):
    while range(0, 20):
        try:
            raw_html = get(url, stream=True).content
        except:
            continue
        html = BeautifulSoup(raw_html, 'html.parser')
        links = html.select('a')
        urls = []
        paths = []
        for i in range(5, len(links)):
            if "512_0193.jpg" in links[i].get('href'):
                print('\t\t\t' + str(i-5) + "/" + str(len(links) - 5) + '\tDAY: ' + str(links[i]))
                urls.append(str(url + links[i].get('href')))
                paths.append(str(links[i].get('href')))
        #ipdb.set_trace()

        pool = eventlet.GreenPool(size=100)
        i = 0
        for result in pool.imap(fetch, urls, paths):
            print(i)
            i += 1
        break

    
def download_days(url):

    while range(0, 20):
        try:
            raw_html = get(url, stream=True).content
        except:
            continue
        html = BeautifulSoup(raw_html, 'html.parser')
        links = html.select('a')
        for i in range(5, len(links)):
            print('\t\tDAYS: ' + str(links[i]))
            if not os.path.exists(str(links[i].get('href'))):
                os.makedirs(str(links[i].get('href')))
            os.chdir(str(links[i].get('href')))
            try:
                download_day(str(url + links[i].get('href')))
            finally:
                os.chdir("..")
        break

def download_months(url):

    while range(0, 20):
        try:
            raw_html = get(url, stream=True).content
        except:
            continue
        html = BeautifulSoup(raw_html, 'html.parser')
        links = html.select('a')
        for i in range(5, len(links)):
            print('\tMONTH: ' + str(links[i]))
            if not os.path.exists(str(links[i].get('href'))):
                os.makedirs(str(links[i].get('href')))
            os.chdir(str(links[i].get('href')))
            try:
                download_days(str(url + links[i].get('href')))
            finally:
                os.chdir("..")
        break


def download_years(years, url):

    raw_html = get(url, stream=True).content
    html = BeautifulSoup(raw_html, 'html.parser')
    links = html.select('a')
    for i in range(5, len(links)):
        if links[i].contents[0][:-1] in years:
            print('YEAR: ' + str(links[i]))
            if not os.path.exists(str(links[i].get('href'))):
                os.makedirs(str(links[i].get('href')))
            os.chdir(str(links[i].get('href')))
            try:
                download_months(str(url + links[i].get('href')))
            finally:
                os.chdir("..")
        
if __name__ == '__main__':

    url = "https://sdo.gsfc.nasa.gov/assets/img/browse/"
    url_2 = "https://sdo.gsfc.nasa.gov/assets/img/browse/2018/01/01/"

    download_years('2018', url)
    #ipdb.set_trace()
    #download_days(url_2)

