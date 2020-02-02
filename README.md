# Capstone_NLTK_SpamClassifier

commands

Install
python3 -m pip install --user --upgrade pip; pip3 install --user -r requirements.txt

Run
export LC_ALL=C.UTF-8; export LANG=C.UTF-8; export FLASK_APP=spamfilter; flask run --host 0.0.0.0 --port 8000

Test
export LC_ALL=C.UTF-8; export LANG=C.UTF-8; python3 -m unittest discover -v > test_report.txt 2>&1