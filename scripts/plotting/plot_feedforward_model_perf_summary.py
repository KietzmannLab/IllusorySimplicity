from os import path
import pandas as pd

from macaquethings.data_util.load_data import get_channel_masks, load_data, load_inter_area_fit_chunks

rectify = False

# ----- INTER AREA
inter_area_dir = path.join('results', 'inter_area')
run_name = f'monkeyF_v1_target_array_{target_array}_stride_2_ridgecv_lfp_hugegrid_avg10ms_allsess'
rundir = path.join(inter_area_dir, run_name)

inter_area_results, recurrent_results, params, cfg = load_inter_area_fit_chunks(rundir)


