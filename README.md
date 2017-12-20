# hltc_preprocess
Preprocessing library for our lab's use

## Setup
`pip install hltc_preprocess`

## Modules
1. vocabulary.py
	- `create_vocabulary`
	- `idx_tokens`
	- `find_phrases`
2. `tweets.py`
	- `clean_tweet`
	- `filter_tweet`
	- `tokenize_tweets`
## How to upload to pip
git tag 0.9
git push --tags origin
python setup.py sdist upload -r pypitest
python setup.py sdist upload -r pypi

http://peterdowns.com/posts/first-time-with-pypi.html
