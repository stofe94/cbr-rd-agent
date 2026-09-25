FROM python:3.11-slim

WORKDIR /root

# System dependencies + Chrome + Docker CLI
RUN apt-get update && apt-get install -y \
    git curl ca-certificates gnupg dos2unix \
    chromium chromium-driver && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/debian bookworm stable" \
        > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install -y docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

# Install deps first (layer cache), then install the package — all into /root
COPY requirements.txt /root/
RUN pip install --no-cache-dir -r /root/requirements.txt

# .git stays out of the image (.dockerignore), so setuptools-scm cannot read the
# version from it; Docker_push_GCR.sh passes the latest release tag.
ARG SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
COPY . /root/
RUN pip install --no-cache-dir --no-deps -e "/root/[dev]"

EXPOSE 19899

CMD ["sh", "-c", "\
    set -o allexport && \
    if [ -f /root/config.env ];   then . /root/config.env;   fi && \
    if [ -f /root/secrets.env ]; then . /root/secrets.env; fi && \
    set +o allexport && \
    exec bash /root/run_linux.sh"]