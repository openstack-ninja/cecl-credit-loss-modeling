#
# virtual environments to avoid conflicts
python3 -m venv .venv
source .venv/bin/activate

#python3 run_cecl_pipeline-ultimate.py --data_dir ./data/fannie --gse fannie --cores 64 --output data/fannie_results.parquet

#python run_cecl_pipeline.py --data_dir ./data/freddie --gse freddie --cores 64 --output freddie_cecl.parquet
python run_cecl_pipeline.py --data_dir ./data/fannie --gse fannie --cores 64 --output data/fannie.parquet
