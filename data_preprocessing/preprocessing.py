import re
from pathlib import Path

import emoji
import pandas as pd
import wordsegment


wordsegment.load()


ROOT = Path(__file__).parent.parent  # repo root folder


# +++++ Hashtags preprocessing +++++ #


_re_hashtags = re.compile(r'(#\w+)')


def hashtag_segmentation(data):
    """segment hashtags"""
    hashtags = _re_hashtags.findall(data)
    for hs in hashtags:
        words = ' '.join(wordsegment.segment(hs))
        data = data.replace(hs, words)
    return data


# +++++ Emojis preprocessing +++++ #


_re_text_emoji = re.compile(r':(\w+):')


def _process_emoji_func(match_obj):
    return match_obj.expand(r'\g<1>').replace('_', ' ')


def emoji_normalization(data):
    """convert emojis to natural language"""
    data = emoji.demojize(data)  # 😢 -> :crying_face:
    data = _re_text_emoji.sub(_process_emoji_func, data)  # process obtained emoji :crying_face: -> crying face
    return data


# +++++ Masks unification +++++ #


_re_md_ref = re.compile(r'\[([^\]]+?)\]\((\S+?)\)')
_re_real_http = re.compile(r'(https?|ftp)://[^\s/$.?#].[^\s]*', flags=re.IGNORECASE+re.MULTILINE)  # https://mathiasbynens.be/demo/url-regex @stephenhay
_re_relative_link = re.compile(r'\/[^\s/$.?#\)]+')
_re_url_masks = re.compile(r'URL|\[URL\]|<URL>|\[http\]')

_re_user_at = re.compile(r'\B@\w{1,32}')
_re_user_masks = re.compile(r'\[USER\]|<USER>|@USER|<MENTION_\d+>|MENTION\d+')

_re_reddit_r = re.compile(r'/?r/([^\s/]+)')
_re_reddit_u = re.compile(r'/?u/[A-Za-z0-9_-]+')

_re_rt = re.compile(r'\bRT\b')


def mask_replacements(data, user_mask: str = 'USER', url_mask: str = 'http'):
    """replace usernames, links, subreddits with tokens + remove special characters"""
    data = _re_md_ref.sub(rf'\g<1> {url_mask}', data)  # replace md links ref

    data = _re_reddit_r.sub(f'subreddit {url_mask}', data)  # replace subreddits with tokens
    data = _re_reddit_u.sub(user_mask, data)  # replace reddit usernames

    data = _re_real_http.sub(url_mask, data)  # replace links with http token
    data = _re_url_masks.sub(url_mask, data)  # replace URL tokens in other datasets with http tokens

    data = _re_user_at.sub(user_mask, data)  # replace tw @handles with USER tokens
    data = _re_user_masks.sub(user_mask, data)  # replace MENTION123, <MENTION_123>, [USER] etc tokens in misogyny datasets with user tokens
    data = re.sub(r'(USER\s*){5,}USER'.replace('USER', user_mask), ' '.join([user_mask] * 5), data)  # replace more than 5 users USER .... USER with 5 users

    data = _re_rt.sub('', data)  # remove tw RT
    return data


# +++++ Spaces normalisation +++++ #


_re_space = re.compile(r'\s+')


def process_spaces(data):
    data = _re_space.sub(' ', data)  # remove double space
    data = data.strip()  # remove whitespace
    return data


# +++++ Select data and process it +++++ #


def preprocess(data: str, do_spaces: bool = True, do_hashtags: bool = False, do_masks: bool = True, do_emoji: bool = True):
    data = str(data)
    if do_masks:
        data = mask_replacements(data)  # replacements
    if do_hashtags:
        data = hashtag_segmentation(data)  # segment hashtags
    if do_emoji:
        data = emoji_normalization(data)  # emoji to nl
    if do_spaces:
        data = process_spaces(data)  # double spaces etc
    return data


def process_file(filename):
    _df = pd.read_csv(filename)  # read csv file
    if 'Unnamed: 0' in _df.columns:
        _df = _df.rename(columns={'Unnamed: 0': 'id'})
    _df['text_preprocessed'] = _df.text.apply(preprocess)  # preprocess
    _df.to_csv(filename, index=False)  # save inplace


if __name__ == '__main__':
    data_folders = [
        ROOT / 'edos_data' / 'processed',
        ROOT / 'multitask_data' / 'formatted',
        ROOT / 'multitask_data' / 'processed',
    ]
    for folder in data_folders:
        print('folder', folder.relative_to(ROOT))
        for filename in folder.glob('**/*.csv'):
            print('  *', filename.relative_to(folder))
            process_file(filename)
