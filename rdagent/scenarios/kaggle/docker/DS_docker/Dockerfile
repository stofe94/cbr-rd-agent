FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    strace \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2t64 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libgdk-pixbuf-2.0-0 \
    libnspr4 \
    libnss3 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    xdg-utils \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub \
    | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
       > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install matching ChromeDriver (Chrome 115+ endpoint)
RUN wget -q "https://storage.googleapis.com/chrome-for-testing-public/$(curl -s https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_STABLE)/linux64/chromedriver-linux64.zip" \
    && unzip chromedriver-linux64.zip \
    && mv chromedriver-linux64/chromedriver /usr/bin/chromedriver \
    && chmod +x /usr/bin/chromedriver \
    && rm -rf chromedriver-linux64.zip chromedriver-linux64

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    xgboost \
    lightgbm \
    catboost \
    matplotlib \
    seaborn \
    pyarrow \
    fastparquet \
    coverage \
    optuna \
    networkx \
    tqdm \
    statsmodels \
    plotly \
    ipython \
    pillow \
    joblib \
    selenium

WORKDIR /workspace