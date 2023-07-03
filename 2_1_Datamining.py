#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time
import math
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
import os.path
import nltk, re, pprint
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize
from nltk.util import ngrams
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

tk = WordPunctTokenizer()
lm = nltk.WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

# Enter Midjourney Discord server
url = 'https://discord.com'
project_dir = './'
cookie_file = project_dir + 'cookies.pkl'

#%% Launch chrome and login to discord
driver = webdriver.Chrome()
action = ActionChains(driver)


def find_element_WAIT(element_identifier, method=By.XPATH, wait=5):
    try:
        WebDriverWait(driver, wait).until(EC.presence_of_element_located((method, element_identifier)))
        return driver.find_element(method, element_identifier)
    except:
        print("ERROR: Element with identifier {} not found".format(element_identifier))
        raise SystemExit

def find_elements_WAIT(element_identifier, method=By.XPATH, wait=5):
    try:
        WebDriverWait(driver, wait).until(EC.presence_of_element_located((method, element_identifier)))
        return driver.find_elements(method, element_identifier)
    except:
        print("ERROR: Elements with identifier {} not found".format(element_identifier))
        raise SystemExit
        
def fix(data):
    link, postid = data
    link_parts = list(link.split('/'))
    if len(link_parts[-1]) > 20:
        channel = link_parts[-1].partition(postid)[0]
        link_parts = list(link_parts[:5])
        link_parts.extend([channel, postid])
        link = '/'.join(link_parts)
    return link

#Query whether post is question. Return bool
def isQuestion(test):
    sentences = nltk.tokenize.sent_tokenize(test)
    for sentence in sentences:
        tokens = tk.tokenize(sentence)
        pos_tags = list(dict(nltk.pos_tag(tokens)).values())
        tagged_bigrams = list(ngrams(pos_tags, 2))
        if '?' in tokens:
            #print("QUESTION: has ?")
            return True
        #elif any(item.startswith('W') for item in pos_tags):
        #    #print("QUESTION: has wh- question pos token")
        #    return True
        elif pos_tags[0].startswith('W'):
            #print("QUESTION: has wh- question pos token")
            return True
        elif any(item == ('MD', 'PRP') for item in tagged_bigrams):
            #print("QUESTION: has a modal-verb pos token bigram")
            return True
        else:
            pass
    return False
        
def DEBUG_show_all_attributes(element):
    html = element.get_attribute('outerHTML')
    attrs = BeautifulSoup(html, 'html.parser').a.attrs
    print(attrs)
    
def login_discord():
    driver.get(url+'/login')
    currentElem = find_element_WAIT('/html/body/div[2]/div[2]/div[1]/div[1]/div/div/div/div/form/div[2]/div/div[1]/div[2]/div[1]/div/div[2]/input')
    currentElem.send_keys('<EMAIL>') #email

    currentElem = driver.find_element(By.XPATH, '/html/body/div[2]/div[2]/div[1]/div[1]/div/div/div/div/form/div[2]/div/div[1]/div[2]/div[2]/div/input')
    currentElem.send_keys('<PASSWORD>') #password
    currentElem.send_keys(Keys.RETURN)


def open_discord():
    try:
        # load cookies for given websites
        #print("Attempting to load cookies...")
        cookies = pickle.load(open(cookie_file, "rb"))
        driver.get(url+'/login')
        for cookie in cookies:
            driver.add_cookie(cookie)
        driver.refresh()
        #print("Loaded cookies successfully!")
    except Exception as e:
        # it'll fail for the first time, when cookie file is not present
        #print(e)
        #print("Attempting to login manually instead...")
        login_discord()
        #print("Logged in successfully!\nSaving cookies...")
        cookies = driver.get_cookies()
        #pickle.dump(cookies, open(cookie_file, "wb"))
        #print("Cookies saved successfully!")

open_discord()

#Enter Midjourney server
currentElem = find_element_WAIT("blobContainer-ikKyFs", By.CLASS_NAME, wait=10)
currentElem.click()



#%% Scrape functions and command (faq's)
"""
The following code scrapes the Midjourney 'prompt-faq'.
This channel has a format different from the chat channels, which includes less info.
As such, the data was scraped, but never used
"""


"""
#Enter 'prompt-faq'
currentElem = find_element_WAIT('prompt-faqs', method=By.LINK_TEXT, wait=10)
currentElem.click()


def parse_date(date):
    TODAY = datetime.date.today()
    splitted = date.split()
    if len(splitted) == 1 and splitted[0].lower() == 'today':
        return str(TODAY.isoformat())
    elif len(splitted) == 1 and splitted[0].lower() == 'yesterday':
        date = TODAY - relativedelta(days=1)
        return str(date.isoformat())
    elif date[1] == 'd':
        date = TODAY - relativedelta(days=int(date[0]))
        return str(date.isoformat())
    else:
        date = datetime.datetime.strptime(date, '%B %d, %Y')
        return str(date.isoformat())

def scrape_n_scroll(channel, fromDate=None, toDate=datetime.date.today().isoformat()):
    data = []
    errors = []
    window_size = driver.find_element(By.TAG_NAME, 'html').rect
    currentUrl = driver.current_url
    try:
        while True:
            time.sleep(1)
            elements = find_elements_WAIT("card-2JNtco",By.CLASS_NAME)
            for i, post in enumerate(elements):
                try:
                    dataId = post.find_element(By.CLASS_NAME, 'container-2qVG6q').get_attribute('data-item-id')
                    dataUrl = str(currentUrl) + '/threads/' + str(dataId)
                    question = post.find_element(By.TAG_NAME, 'h3').text
                    comments = post.find_element(By.CLASS_NAME, 'message-1zs6Od').text
                    try:
                        reactions = post.find_element(By.CLASS_NAME, 'reactionCount-SWXh9W').text
                    except:
                        reactions = 0
                    try:
                        messages = post.find_element(By.CLASS_NAME, 'messageCountText-J-9pag').text
                    except:
                        messages = 0
                    date = parse_date(post.find_element(By.CLASS_NAME, 'text-sm-normal-AEQz4v').text)
                    #print('{}\n{}'.format(question, date))
                    #posts[question] = [date, comments]
                    data.append([dataId, dataUrl, question, comments, date, reactions, messages])
                    if date[:-9] == toDate or date[:-9] == datetime.date.today().isoformat()[:-9]:
                        with open(project_dir + channel + '-list.pkl', "wb") as output_file:
                            pickle.dump(data, output_file)
                        df_faq = pd.DataFrame(data, columns=['post id',
                                                             'link to post',
                                                             'post',
                                                             'comments',
                                                             'date posted',
                                                             'num. reactions',
                                                             'num. messages'])
                        
                        df_faq = df_faq.drop_duplicates(inplace=True)
                        with open(project_dir + channel + '-dataframe.pkl', "wb") as output_file:
                            pickle.dump(df_faq, output_file)
                        with open(project_dir + channel + '-ERROR.pkl', "wb") as output_file:
                            pickle.dump(errors, output_file)
                        return data, df_faq, errors
                    elif len(data) % 100 == 0:
                        with open(project_dir + channel + '-partial.pkl', "wb") as output_file:
                            pickle.dump(data, output_file)
                    else:
                        pass
                except Exception as E:
                    #print('ERROR: question not found on element {}'.format(post))
                    print(E)
                    errors.append(E)
                   # pass
            # Scroll down to bottom card and refresh
            action.move_to_element(driver.find_element(By.TAG_NAME, 'html')).move_by_offset(math.floor(window_size['width']/2)-1, math.floor(window_size['height']/2)-1).click().perform() #50 worked
            #action.move_to_element(elements[-1]).perform() #50 worked
    except KeyboardInterrupt:
        with open(project_dir + channel + '-partial.pkl', "wb") as output_file:
            pickle.dump(data, output_file)
        with open(project_dir + channel + '-ERROR.pkl', "wb") as output_file:
            pickle.dump(errors, output_file)
            
time.sleep(5)

data, dataframe, errors = scrape_n_scroll('prompt-tips', toDate='2022-09-20') """

#%% Scrape functions (chat)


def parse_date2(date):
    TODAY = datetime.date.today()
    splitted = date.split()
    if splitted[0].lower() == 'today':
        splitted[0] = str(TODAY)
        return str(datetime.datetime.strptime(' '.join(splitted), '%Y-%m-%d at %I:%M %p').isoformat())
    elif splitted[0].lower() == 'yesterday':
        date = str(TODAY - relativedelta(days=1))
        splitted[0] = date
        return str(datetime.datetime.strptime(' '.join(splitted), '%Y-%m-%d at %I:%M %p').isoformat())
    elif '-' in date:
        date = datetime.datetime.strptime(date, '%Y-%m-%d').isoformat()
        return str(date)
    else:
        date = datetime.datetime.strptime(date, '%m/%d/%Y %I:%M %p').isoformat()
        return str(date)


def scrape_n_scroll2(channel, fromDate=None, toDate=datetime.date.today().isoformat()):
    try:
        if fromDate == None:
            #currentElem = find_element_WAIT('#app-mount > div.appAsidePanelWrapper-ev4hlp > div.notAppAsidePanel-3yzkgB > div.app-3xd6d0 > div > div.layers-OrUESM.layers-1YQhyW > div > div > div > div.content-1SgpWY > div.chat-2ZfjoI > section > div.toolbar-3_r2xA > div:nth-child(3)', By.CSS_SELECTOR)
            #currentElem = find_element_WAIT('[arial-label^=Pinned]', By.CSS_SELECTOR)
            #currentElem.click()
    
            #elements = find_elements_WAIT("messageGroupWrapper-1jf_7C",By.CLASS_NAME)
            #action.move_to_element(elements[-1]).perform()
            #time.sleep(1)
            #elements[-1].find_element(By.CLASS_NAME,'jumpButton-1ZwI_j').click()
            search = find_element_WAIT('searchBar-jGtisZ', By.CLASS_NAME)
            action.move_to_element(search).click().send_keys('in:{}'.format(channel)).move_by_offset(0,70).click().move_to_element(search).perform()
            #action.move_to_element(search).click().send_keys('in:{} during:{}'.format(channel,fromDate)).send_keys(Keys.RETURN).perform()
            #action.move_to_element(search).click().send_keys('in:{}'.format(channel)).pause(1).send_keys('during:{}'.format(fromDate)).send_keys(Keys.RETURN).perform()
            time.sleep(8)
            currentElem = find_element_WAIT('#app-mount > div.appAsidePanelWrapper-ev4hlp > div.notAppAsidePanel-3yzkgB > div.app-3xd6d0 > div > div.layers-OrUESM.layers-1YQhyW > div > div > div > div.content-1SgpWY > div.chat-2ZfjoI > div.content-1jQy2l > section > header > div.searchHeaderTabList-3CZMQB.side-1lrxIh > div:nth-child(2)',By.CSS_SELECTOR).click()
            currentElem = find_elements_WAIT('container-rZM65Y', By.CLASS_NAME, wait=8)
            action.move_to_element(currentElem[0]).perform()
            action.move_to_element(currentElem[0].find_element(By.CLASS_NAME, 'buttonsContainer-Nmgy7x')).click().perform()
            search = find_element_WAIT('searchBar-jGtisZ', By.CLASS_NAME)
            search.find_element(By.CLASS_NAME, 'iconContainer-1RqWJj').click()
            data = []
        else:
            try:
                with open(project_dir + channel +'-partial-'+str(fromDate)+'-'+str(toDate)+'.pkl', "rb") as input_file:
                   data = pickle.load(input_file)
                lastLink = data[-1][3]
                driver.get(lastLink)
            except:
                data = []
                search = find_element_WAIT('searchBar-jGtisZ', By.CLASS_NAME)
                action.move_to_element(search).click().send_keys('during:{} in:{}'.format(fromDate, channel)).move_by_offset(0,70).click().move_to_element(search).perform()
                currentElem = find_element_WAIT('#app-mount > div.appAsidePanelWrapper-ev4hlp > div.notAppAsidePanel-3yzkgB > div.app-3xd6d0 > div > div.layers-OrUESM.layers-1YQhyW > div > div > div > div.content-1SgpWY > div.chat-2ZfjoI > div.content-1jQy2l > section > header > div.searchHeaderTabList-3CZMQB.side-1lrxIh > div:nth-child(2)',By.CSS_SELECTOR, wait=8).click()
                currentElem = find_elements_WAIT('container-rZM65Y', By.CLASS_NAME, wait=8)
                action.move_to_element(currentElem[0]).perform()
                time.sleep(1)
                action.move_to_element(currentElem[0].find_element(By.CLASS_NAME, 'buttonsContainer-Nmgy7x')).click().perform()
                search = find_element_WAIT('searchBar-jGtisZ', By.CLASS_NAME)
                search.find_element(By.CLASS_NAME, 'iconContainer-1RqWJj').click()
        old_date = None
        errors = []
        #window_size = driver.find_element(By.TAG_NAME, 'html').rect
        currentUrl = driver.current_url
        server = find_element_WAIT('headerContent-2SNbie', By.CLASS_NAME).text
        action.move_to_element(find_element_WAIT('chatContent-3KubbW', By.CLASS_NAME)).perform()
        while True:
            time.sleep(1)
            #elements = find_elements_WAIT("message-2CShn3",By.CLASS_NAME)
            #action.move_to_element(find_element_WAIT('chatContent-3KubbW', By.CLASS_NAME)).perform()
            #elements = find_elements_WAIT('[id^=chat-messages-]', By.CSS_SELECTOR)
            try:
                elements = [x for x in find_elements_WAIT('message-2CShn3', By.CLASS_NAME) if x not in set(elements)]
            except:
                elements = find_elements_WAIT('message-2CShn3', By.CLASS_NAME)
            for post in elements:
                try:
                    dataId = str(post.get_attribute('data-list-item-id'))[-19:]
                    dataId = dataId.replace('-','')
                    if any(dataId in posts for posts in data):
                        continue
                    dataUrl = str(currentUrl[:-len(dataId)]) + str(dataId)
                    dataUrl = fix((dataUrl,dataId))
                    try:
                        username = post.find_element(By.ID, str('message-username-'+dataId)).text
                    except:
                        try:
                            username = post.find_element(By.CLASS_NAME, 'username-3_PJ5r').text
                        except:
                            oldID = post.get_attribute('aria-labelledby').split(' ')[0].split('-')[-1]
                            link_parts = dataUrl.split('/')
                            prevpost = find_element_WAIT('//*[@id="chat-messages-{}-{}"]'.format(link_parts[-2], oldID), By.XPATH)
                            username = prevpost.find_element(By.ID, str('message-username-'+oldID)).text
                    content = post.find_element(By.ID, str('message-content-'+dataId)).text
                    question = isQuestion(content)
                    try:
                        polarity_compoundScores = []
                        for sentence in sent_tokenize(content):
                            ValenceScores_dict = analyzer.polarity_scores(sentence)
                            polarity_compoundScores.append(ValenceScores_dict['compound'])
                        valence = np.mean(polarity_compoundScores)
                    except:
                        valence = 0
                    try:
                        reply = post.find_element(By.CLASS_NAME,'repliedTextContent-2hOYMB').get_attribute('id').split('message-content-')[-1]
                    except:
                        reply = np.nan
                    try:
                        reactions = post.find_element(By.CLASS_NAME, 'reactions-3ryImn').text
                        reactions = np.sum(np.array(reactions.split(), dtype=np.int))
                    except:
                        reactions = 0
                    """
                    try:
                        pictureUrl = post.find_elements(By.CLASS_NAME, 'originalLink-Azwuo9')
                        print(pictureUrl)
                        pictureUrl = [ str(x.get_attribute('href')) for x in pictureUrl ]
                        print(pictureUrl)
                        pictureUrl = ', '.join(pictureUrl)
                        print(pictureUrl)
                        break
                        #picture = str(post.find_element(By.CLASS_NAME, 'originalLink-Azwuo9').get_attribute('href'))
                    except Exception as E:
                        print(E)
                        pictureUrl = ''
                    """
                    try:
                        #date = post.find_element(By.XPATH, '#div > div.contents-2MsGLg > h3 > span.timestamp-p1Df1m.timestampInline-_lS3aK').text
                        #date = post.find_element(By.TAG_NAME, 'time').text
                        date = post.find_element(By.CLASS_NAME, 'timestamp-p1Df1m').text
                        date = parse_date2(date)
                        if fromDate == None:
                            fromDate = date[:-9]
                        else:
                            pass
                    except:
                        date = old_date
                    data.append([server, channel, dataId, dataUrl, username, content, question, reply, date, reactions, valence])
                    old_date = date
                    #print(data)
                    if date[:-9] == parse_date2(toDate)[:-9] or date[:-9] == datetime.date.today().isoformat()[:-9]:
                        with open(project_dir + channel +'-list-'+str(fromDate)+'-'+str(toDate)+'.pkl', "wb") as output_file:
                            pickle.dump(data, output_file)
                        dataframe = pd.DataFrame(data, columns=['server',
                                                             'channel',
                                                             'post id',
                                                             'link to post',
                                                             'username',
                                                             'post',
                                                             'is question',
                                                             'reply to id',
                                                             'date posted',
                                                             'num. reactions',
                                                             'valence'])
                                                             #'links to images'])
                        dataframe = dataframe.drop_duplicates().reset_index(drop=True)
                        with open(project_dir + channel +'-dataframe-'+str(fromDate)+'-'+str(toDate)+'.pkl', "wb") as output_file:
                            pickle.dump(dataframe, output_file)
                        with open(project_dir + channel +'-ERROR-'+str(fromDate)+'-'+str(toDate)+'.pkl', "wb") as output_file:
                            pickle.dump(errors, output_file)
                        return data, dataframe, errors
                    elif len(data) % 100 == 0:
                        print(date)
                        with open(project_dir + channel +'-partial-'+str(fromDate)+'-'+str(toDate)+'.pkl', "wb") as output_file:
                            pickle.dump(data, output_file)
                    else:
                        pass
                except Exception as E:
                    #print('ERROR: question not found on element {}'.format(post))
                    print(E)
                    errors.append(str(E))
                    with open(project_dir + channel +'-ERROR-'+str(fromDate)+'-'+str(toDate)+'.pkl', "wb") as output_file:
                        pickle.dump(errors, output_file)
            action.move_to_element(post).perform()
        return False
    except KeyboardInterrupt:
        with open(project_dir + channel +'-partial-'+str(fromDate)+'-'+str(toDate)+'.pkl', "wb") as output_file:
            pickle.dump(data, output_file)
        with open(project_dir + channel +'-ERROR-'+str(fromDate)+'-'+str(toDate)+'.pkl', "wb") as output_file:
            pickle.dump(errors, output_file)
        return False

#%% Scrape channels
server_list = driver.find_element(By.XPATH, '//*[@id="app-mount"]/div[2]/div[1]/div[1]/div/div[2]/div/div/nav/ul/div[2]/div[3]')
servers = server_list.find_elements(By.CLASS_NAME, 'listItem-3SmSlK')
channels = {'Midjourney': ['prompt-chat'],
            'OpenAI': ['dall-e-discussions',
                       'de2-prompt-help',
                       'de2-tips-n-tricks'],
            'Stable Foundation': ['general-chat',
                                  'prompting-help',
                                  'sd-chat'],
            'r/StableDiffusion': ['prompting',
                                  'textual-inversion']}
fromDate = None
toDate = '2023-04-15'

for server in servers:
    server.click()
    server_name = find_element_WAIT('headerContent-2SNbie', By.CLASS_NAME).text
    for channel in channels[server_name]:
        print(channel)
        if channel+'-dataframe-'+str(fromDate)+'-'+str(toDate)+'.pkl' not in os.listdir(project_dir):
            #try running twice. Sometimes that seems to help
            try:
                data, dataframe, errors = scrape_n_scroll2(channel, fromDate, toDate)
            except Exception as E:
                print('ERROR: {} failed to run with message {}'.format(channel, E))
        else:
            continue

#%% Traverse  project folder to get all dataframes (except faqs) 
def traverseFolder(path, skip=[]):
    dataframes = []
    for filename in os.listdir(path):
        #don't fetch non-dataframe objects
        #don't fetch any files marked as 'skip'
        if 'dataframe' in filename and not any(channel in filename for channel in skip):
            print(filename)
            with open(project_dir + filename, "rb") as input_file:
              dataframe = pickle.load(input_file)
            dataframes.append(dataframe)
    master_dataframe = pd.concat(dataframes)
    master_dataframe = master_dataframe.drop_duplicates().reset_index(drop=True)
    master_dataframe['date posted'] = pd.to_datetime(master_dataframe['date posted'],infer_datetime_format=True)
    return master_dataframe

MASTER = traverseFolder(project_dir+'Data/', ['prompt-faq', 'prompt-tips'])

with open(project_dir + 'MASTER_all_data_raw.pkl', "wb") as output_file:
  pickle.dump(MASTER, output_file)