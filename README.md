# Semantic Information retrieval

## Installation Steps
Install ElasticSearch
```commandline
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
sudo apt-get install apt-transport-https
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-7.x.list
sudo apt-get update && sudo apt-get install elasticsearch
```

Install package requirements in an virtual env.
```commandline
pip install -r retrieval/requirements.txt
```

Install the `retrieval` package in the current env. in edition mode
```commandline
pip install -e .
```

## Running
Configure Elastic setting by modifying `retrieval/retrieval/settings.py`

Set Elasticsearch host, port, user, password, max retrieve and insert params.

Package is a Flack App.

Move to `retrieval/retrieval` folder and run
```commandline
python app.py --index [0|1] --dataset [squad|sweez]
```

If `index` is 0, the app will be launched without running indexing of the `dataset.

The server is ready to handle query on port `5000`.

```commandline
CURL -XGET "http://127.0.0.1:5000/query?query=What is the Neaming of Named Entities&size=10"
```
