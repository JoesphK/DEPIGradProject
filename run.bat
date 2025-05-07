@echo off
echo Activating the virtual environment...
call hc\Scripts\activate

echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
@echo Starting the Flask API...
python run_api.py

pause
