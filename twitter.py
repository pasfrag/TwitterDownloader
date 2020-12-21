import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
import os
import shutil
from mongo import MongoHandler
from secret_keys import consumer_key, consumer_secret, access_token_secret, access_token, model_path, meme_path, non_meme_path
from tweepy import API, OAuthHandler, Cursor
import cv2
from urllib.request import urlopen
import numpy as np


class TweetMiner(object):
    api = None
    connection = None
    model = None

    def __init__(self):
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        self.connection = MongoHandler()
        self.model = load_model(model_path)

    def get_new_tweets(self):
        for tweet in Cursor(self.api.search, q="#brexit filter:images", lang="en", tweet_mode="extended", count=5,
                            include_entities=True).items():
            if not tweet._json["full_text"].startswith("RT @") and \
                    ("promo" or "giveaway") not in tweet._json["full_text"] and \
                    len(tweet._json["full_text"].split()) >= 5:
                dct_to_save = tweet._json
                label = self.image_prediction(dct_to_save['entities']['media'][0]['media_url'])

                if label != -1:
                    dct_to_save['_id'] = dct_to_save.pop('id')
                    dct_to_save['label'] = label
                    self.connection.store_to_collection(dct_to_save, 'twitter')

                self.clean_data_folder()

    # Downloads image without saving
    def url_to_image(self, url):
        resp = urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        cv2.imwrite('data/' + url.split('/')[-1], image)

    # Predicting image
    def image_prediction(self, url):

        # TODO add text detection, if image has no text return -1

        path_to_image = ''
        for filename in os.listdir('data/'):
            path_to_image = 'data/' + filename

        img = load_img(
            path_to_image, target_size=(240, 240)
        )
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        label = np.argmax(score)
        return label

    # Delete data
    def clean_data_folder(self):
        for filename in os.listdir('data/'):
            file_path = 'data/' + filename
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    # TODO add text detection, if image has no text throw it
    # TODO move image to corresponding class folder, return path to folder and add to dict
