# Daily startup 
cd C:\Users\aylia\VScode\ai-smart-walker

venv\Scripts\activate

# First time only (or after pulling new changes) 
WINDOWS: pip install -r requirements.txt

PI: pip install -r requirements-pi.txt

# When you install new package 
pip freeze > requirements.txt
