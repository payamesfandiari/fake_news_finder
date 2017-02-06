
import pandas as pd
from os import path
from PIL import Image
import numpy as np
import random
from palettable.colorbrewer.sequential import Reds_9
from palettable.colorbrewer.sequential import Greens_9
from wordcloud import WordCloud, STOPWORDS


def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return tuple(Reds_9.colors[random.randint(2,8)])

def color_func_green(word, font_size, position, orientation, random_state=None, **kwargs):
    return tuple(Greens_9.colors[random.randint(2,8)])
#
# d = path.dirname(__file__)
fa_path = "exported/"
font_path = "OpenSans-CondBold.ttf"
#
# icon = "thumbs-down"
#
# icon_path = fa_path + "%s.png" % icon
# icon = Image.open(icon_path)
# mask = Image.new("RGB", icon.size, (255, 255, 255))
# mask.paste(icon, icon)
# mask = np.array(mask)
#
# stopwords = set(STOPWORDS)
# stopwords.add("said")
#
# wc = WordCloud(font_path=font_path, background_color="white", max_words=2000, mask=mask,
#                max_font_size=300, stopwords=stopwords)
#
#
#
# # Read the whole text.
# fakes = pd.read_csv('clean_fake_data.csv',encoding='latin1',index_col=0)
# fakes = fakes.dropna()
# text = ''.join(fakes['text'].values.tolist())
#
#
#
# # generate word cloud
# wc.generate(text)
# wc.recolor(color_func=color_func, random_state=3)
# wc.to_file("fake_neg_wordcloud.png")


####======================= Generating for the True Class ===================

icon = "thumbs-up"

icon_path = fa_path + "%s.png" % icon
icon = Image.open(icon_path)
mask = Image.new("RGB", icon.size, (255, 255, 255))
mask.paste(icon, icon)
mask = np.array(mask)

stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(font_path=font_path, background_color="white", max_words=2000, mask=mask,
               max_font_size=300, stopwords=stopwords)



# Read the whole text.
reals = pd.read_csv('clean_true_data.csv',encoding='latin1')
reals = reals.dropna()
text = ''.join(reals['text'].values.tolist())



# generate word cloud
wc.generate(text)
wc.recolor(color_func=color_func_green)
wc.to_file("real_pos_wordcloud.png")





