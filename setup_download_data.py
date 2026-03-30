#
# virtual environments to avoid conflicts
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install --upgrade playwright 
playwright install

##
export FANNIE_MAE_EMAIL=mimic-hushes-03@icloud.com
export FANNIE_MAE_PASSWORD=diqji3-ziwsis-hivzYq

