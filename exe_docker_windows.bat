@echo off
setlocal

set GITHUB_USER=stofe94
set IMAGE_REMOTE=ghcr.io/%GITHUB_USER%/local_cbr-rdagent:latest
set IMAGE_LOCAL=local_cbr-rdagent:latest
set WIN_WS=%USERPROFILE%\cbr-rdagent
set WIN_TMP=%USERPROFILE%\tmp
set REPO_DIR=%~dp0
rem Docker needs the host workspace with forward slashes
set HOST_WS=%WIN_WS:\=/%/workspace


set /p RD_COMPETITION=Which competition to run? (e.g. spaceship-titanic):

echo [0/3] Ensuring folders exist...
if not exist "%WIN_WS%\workspace\knowledge_base" mkdir "%WIN_WS%\workspace\knowledge_base"
if not exist "%WIN_WS%\log" mkdir "%WIN_WS%\log"
if not exist "%WIN_WS%\log\terminal.log" type nul > "%WIN_WS%\log\terminal.log"

rem Read KEY=value lines from secrets.env, skipping comment lines
for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%WIN_TMP%\secrets.env") do set "%%a=%%b"

rem The pre-built image on ghcr.io is private (maintainer access via GHCR_TOKEN).
rem Without access, the image is built locally from this repository.
echo [1/3] Getting the image...
if "%GHCR_TOKEN%"=="" goto :build
echo %GHCR_TOKEN%| docker login ghcr.io -u %GITHUB_USER% --password-stdin >nul 2>&1 || goto :build
docker pull %IMAGE_REMOTE% || goto :build
docker tag %IMAGE_REMOTE% %IMAGE_LOCAL% || goto :error
goto :start

:build
echo No access to %IMAGE_REMOTE%; building %IMAGE_LOCAL% from %REPO_DIR%...
set VERSION=
for /f "delims=" %%v in ('git -C "%REPO_DIR%." describe --tags --abbrev^=0 2^>nul') do set VERSION=%%v
if defined VERSION (set VERSION=%VERSION:v=%) else (set VERSION=0.0.0)
docker build --build-arg SETUPTOOLS_SCM_PRETEND_VERSION=%VERSION% -t %IMAGE_LOCAL% "%REPO_DIR%." || goto :error

:start
echo [2/3] Removing a previous container...
docker rm -f cbr-rdagent 2>nul

echo [3/3] Starting container...
    docker run -it ^
    -p 19899:19899 ^
    -v "%WIN_WS%\workspace:/root/workspace" ^
    -v "%WIN_WS%\log:/root/log" ^
    -v "%WIN_WS%\log\terminal.log:/root/terminal.log" ^
    -v //./pipe/docker_engine://./pipe/docker_engine ^
    -v "%WIN_TMP%\config.env:/root/config.env:ro" ^
    -v "%WIN_TMP%\secrets.env:/root/secrets.env:ro" ^
    -e DOCKER_HOST=tcp://host.docker.internal:2375 ^
    -e DS_LOCAL_DATA_PATH=/root/workspace ^
    -e HOST_WORKSPACE=%HOST_WS% ^
    --entrypoint bash ^
    --name cbr-rdagent ^
    %IMAGE_LOCAL% ^
    -c "sed 's/--competition [^ ]*/--competition %RD_COMPETITION%/' /root/run_linux.sh | bash"

goto :eof

:error
echo.
echo [ERROR] Script failed. Exit code: %errorlevel%
exit /b %errorlevel%
