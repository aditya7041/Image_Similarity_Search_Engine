#!/usr/local/bin/python
# coding=utf-8

from bs4 import BeautifulSoup
from base64 import decodestring
from urlparse import urljoin

import requests
import urllib2
import os, sys, traceback
import argparse

def create_path_if_not_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir): #Race condition has to be handled
        print 'Create directory ' + dir
        os.makedirs(dir)

def get_amazon_shopping_page(search_query):
    url = 'https://www.amazon.com/s/ref=nb_sb_noss_1?url=search-alias%3Daps'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/56.0.2924.87 Safari/537.36'
    }
    params = {'field-keywords': search_query}
    response = requests.get(url, params=params, headers=headers)

    return response.text


def save_amz_images_links(page_content, path):
    # use Beautiful soap to parse the html
    html = BeautifulSoup(page_content, 'lxml')

    base_url = 'https://www.amazon.com'
    img_no = 1

    # create target directory, if it is not already existing
    create_path_if_not_exists(path)

    with open(os.path.join(path, 'image_links.txt'), 'w') as links_file:

        # iterate on all results
        #retrive all li tags with data-asin attribute
        for tag in html.find_all('li', attrs={'data-asin': True}):

            image_tag = tag.find('img')
            # if there is no image tag or parent doesn't have href attribute, then ignore
            if image_tag is None or not image_tag.parent.has_attr('href'):
                continue

            # print str(img_no)

            # Extract image url
            buy_link = urljoin(base_url, image_tag.parent['href'])
            image_url = image_tag['src']

            img_file_name = os.path.join(path, str(img_no) + '.jpg')

            with open(img_file_name, 'wb') as img_file:
                img_data = requests.get(image_url).content
                img_file.write(img_data)

            links_file.write(buy_link + '\n')

            img_no = img_no + 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download Google images')
    parser.add_argument('-Q', '--query', help='Search term', required=True)
    parser.add_argument('-T', '--target', help='Location to store image files', required=True)

    args = parser.parse_args()

    try:
        page_content = get_amazon_shopping_page(args.query)
        open(os.path.join(args.target, 'amz_response.html'), 'w').write(page_content.encode('utf-8'))
        save_amz_images_links(page_content, args.target)

    except: # catch *all* exceptions
        traceback.print_exc(file=sys.stdout)



