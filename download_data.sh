# download_data.sh

HOME_DIR=$(dirname "$0")
DATA_DIR="$HOME_DIR/.env/data"

if [[ ! -d $DATA_DIR ]]; then
  mkdir "$DATA_DIR"
fi

cd "$DATA_DIR" || exit

if [[ ! -d 'wiki_zh' ]]; then
  wget https://storage.googleapis.com/nlp_chinese_corpus/wiki_zh_2019.zip
  unzip -q wiki_zh_2019.zip
fi

if [[ ! -f 'NP_score.pkl.gz' ]]; then
  wget https://github.com/gablg1/ORGAN/raw/master/organ/NP_score.pkl.gz
fi

if [[ ! -f 'SA_score.pkl.gz' ]]; then
  wget https://github.com/gablg1/ORGAN/raw/master/organ/SA_score.pkl.gz
fi
