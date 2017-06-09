cd Data
bash clear_processed_data.sh
cd ../data_preparation
python process_data.py
cd ../fmri_analysis/utils
python move_EVs.py