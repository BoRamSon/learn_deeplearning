@echo off
echo 🏭 제조업 안전사고 감지 시스템 - Streamlit 데모 시작
echo.

REM 현재 디렉터리를 safety_demo로 변경
cd /d "%~dp0"

REM uv 환경에서 streamlit 실행
echo Streamlit 앱을 시작합니다...
echo 브라우저에서 http://localhost:8501 로 접속하세요.
echo.
echo 종료하려면 Ctrl+C를 누르세요.
echo.

uv run streamlit run app.py

pause
