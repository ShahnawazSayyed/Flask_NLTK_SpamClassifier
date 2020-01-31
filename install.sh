mkdir instance && cd instance && mv ../copyy.py config.py && cd ..
cd spamfilter && mkdir inputdata && mkdir mlmodels && cd ..
mkdir tests/data/inputdata && cp tests/data/sample_emails.csv tests/data/inputdata && cp tests/data/sample_email3.txt tests/data/inputdata
mkdir tests/data/mlmodels && cp tests/data/sample_emails.pk tests/data/mlmodels && cp tests/data/sample_emails_word_features.pk tests/data/mlmodels