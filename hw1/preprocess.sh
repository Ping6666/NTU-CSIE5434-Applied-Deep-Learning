if [ ! -f glove.840B.300d.zip ] && [ ! -f glove.840B.300d.txt ]; then
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove.840B.300d.zip
fi

if [ ! -f glove.840B.300d.txt ]; then
  unzip glove.840B.300d.zip
fi

python ./src/preprocess_intent.py
python ./src/preprocess_slot.py
