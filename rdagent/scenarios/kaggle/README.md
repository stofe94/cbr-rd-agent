# Kaggle Crawler

Chrome is only needed when a competition folder has no `description.md`; the
crawler then reads the description from the Kaggle page. MLE-bench data brings the
file, and the Docker image contains Chromium.

## Install chrome & chromedriver for Linux

In one folder
```shell
# install chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb
google-chrome --version

# install chromedriver
wget https://storage.googleapis.com/chrome-for-testing-public/<chrome-version>/linux64/chromedriver-linux64.zip
unzip chromedriver-linux64.zip
cd chromedriver-linux64
sudo mv chromedriver /usr/local/bin
sudo chmod +x /usr/local/bin/chromedriver

chromedriver --version
```

## config

1. authentication: `KAGGLE_USERNAME` and `KAGGLE_KEY` in `secrets.env` (or `~/.kaggle/kaggle.json`)
2. Accept Rules in competition website. (Join Competition)

## notebook crawler

1. `download_notebooks()`
2. `convert_notebooks_to_text()`