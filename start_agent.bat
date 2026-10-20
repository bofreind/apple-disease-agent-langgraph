@echo off
title Apple Pest QA Agent (LangGraph + PostgreSQL)
echo ==========================================
echo   Apple Pest QA Agent - LangGraph Edition
echo ==========================================
echo.

REM Qwen (DashScope) is a domestic service. Clear proxy vars so the LLM is reachable.
REM NOTE: keep this file ASCII-only. cmd.exe parses .bat in the console codepage
REM (GBK on zh-CN), so UTF-8 Chinese comments here break the script.
set ALL_PROXY=
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=

set PY=D:\anaconda\envs\langchain1-2study\python.exe
set DIR=%~dp0

echo [1/3] Ensuring PostgreSQL checkpoint tables ...
pushd "%DIR%"
"%PY%" init_db.py
if errorlevel 1 (
    echo [ERROR] init_db failed. Is PostgreSQL running?
    echo         docker compose -f docker-compose.pg.yml up -d
    popd
    pause
    exit /b 1
)

echo.
echo [2/3] Self test ...
"%PY%" selftest.py

echo.
echo [3/3] Starting agent service on :8002 ...
echo       Swagger UI: http://localhost:8002/docs
"%PY%" app.py
popd
pause
