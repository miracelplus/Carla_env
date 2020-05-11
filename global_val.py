from tensorboardX import SummaryWriter
from bidict import bidict
import os
import numpy as np
import scipy.io
import pickle

# =========== Simulation parameters ==========
simulation_resolution = 1  # The time interval(resolution) of the simulation platform (second)
# =========== Vehicle parameters =============
LENGTH = 5
# =========== Highway parameters =============
HIGHWAY_LENGTH = 1200  # 1200 1600
EXIT_LENGTH = 800  # 800 1400
v_min, v_max = 20, 40  # m/s vehicles speed limit
a_min, a_max = -4, 2

# =========== Initialization of env ===========
random_initialization_vehicles_count = 18
random_initialization_BV_v_min, random_initialization_BV_v_max = 28, 32
random_env_BV_generation_range_min, random_env_BV_generation_range_max = 0, 300  # m generate BV in [0,200]
initial_CAV_speed=30   # 28m/s
initial_CAV_position = 400  # 200
path1 = os.path.abspath('.')
Initialization_CF_presum_array = np.load(path1 + "/Data/NDD_DATA/Initialization_CF_presum_array_IVBSS_with_safety_guard.npy")
with open(path1 + '/Data/NDD_DATA/Initialization/presum_list_forward.pkl', 'rb') as f:
    # print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT!")
    presum_list_forward = pickle.load(f)
    
# =========== NDD information ===========
CF_percent = 0.6823141  # CF/CF+FF 0.682314106314905
ff_dis = 100  #  dis for ff
gen_length = 1200 # 1200 1600 # stop generate BV after this length
random_veh_pos_buffer_start, random_veh_pos_buffer_end = 0, 50
bv_obs_range = 100 # BV obs range, should consistent with ff distance
round_rule = "Round_to_closest" # Round_to_closest/ Default is r, rr round down, v, acc round up


speed_CDF = list(np.load(path1 + "/Data/NDD_DATA/speed_CDF.npy"))
CF_pdf_array = np.load(path1 + "/Data/NDD_DATA/New/CF_pdf_array_IVBSS_after_preprocess_fully_SG.npy")  # CF_pdf_array_IVBSS_after_preprocess_fully_SG CF_pdf_array_IVBSS_after_preprocess_KDE_Gau_0.5
FF_pdf_array = np.load(path1 + "/Data/NDD_DATA/FF_pdf_array.npy")
SLC_pdf = np.load(path1 + "/Data/NDD_DATA/Xintao_process_NDD/10119_10630_SLC_pdf_smoothed.npy")  # "/Data/NDD_DATA/New/Single_LC_pdf_1_stage_SG.npy  10119_10630_SLC_pdf
DLC_pdf = np.load(path1 + "/Data/NDD_DATA/Xintao_process_NDD/10119_10630_DLC_pdf_smoothed.npy")  # "/Data/NDD_DATA/New/Double_LC_pdf_one_stage_SG.npy" 
OL_pdf = np.load(path1 + "/Data/NDD_DATA/Xintao_process_NDD/10106_10620_OL_pdf_smoothed.npy")  # "/Data/NDD_DATA/New/One_lead_LC_pdf.npy"  /Data/NDD_DATA/Xintao_process_NDD/10106_10620_OL_pdf.npy
CI_pdf = np.load(path1 + "/Data/NDD_DATA/Xintao_process_NDD/10119_10630_CI_pdf_smoothed.npy")
print("================Load Action NDD data finished!=================")

# NDD CF/FF
r_low, r_high, rr_low, rr_high, v_low, v_high, acc_low, acc_high = 0, 115, -10, 8, 20, 40, -4, 2
r_step, rr_step, v_step, acc_step = 1, 1, 1, 0.2
r_to_idx_dic, rr_to_idx_dic, v_to_idx_dic, v_back_to_idx_dic, acc_to_idx_dic = bidict(), bidict(), bidict(), bidict(), bidict()  # Two way dictionary, the front path is: key: real value, value: idx
speed_list, r_list, rr_list = list(range(v_low, v_high+v_step, v_step)), list(range(r_low, r_high+r_step, r_step)), list(range(rr_low, rr_high+rr_step,rr_step)) # used to round speed
num_r, num_rr, num_v, num_acc = CF_pdf_array.shape
for i in range(num_r): r_to_idx_dic[list(range(r_low, r_high+r_step, r_step))[i]] = i
for j in range(num_rr): rr_to_idx_dic[list(range(rr_low, rr_high+rr_step,rr_step))[j]] = j
for k in range(num_v): v_to_idx_dic[list(range(v_low, v_high+v_step, v_step))[k]] = k
for m in range(num_acc): acc_to_idx_dic[list(np.arange(acc_low, acc_high+acc_step, acc_step))[m]] = m

# NDD One lead
one_lead_v_to_idx_dic, one_lead_r_to_idx_dic, one_lead_rr_to_idx_dic = bidict(), bidict(), bidict()
one_lead_r_low, one_lead_r_high, one_lead_rr_low, one_lead_rr_high, one_lead_v_low, one_lead_v_high = 0, 115, -10, 8, 20, 40
one_lead_r_step, one_lead_rr_step, one_lead_v_step = 1, 1, 1
one_lead_speed_list, one_lead_r_list, one_lead_rr_list = list(range(one_lead_v_low, one_lead_v_high+one_lead_v_step, one_lead_v_step)), list(range(one_lead_r_low, one_lead_r_high+one_lead_r_step, one_lead_r_step)), list(range(one_lead_rr_low, one_lead_rr_high+one_lead_rr_step,one_lead_rr_step))
for i in range(int((one_lead_v_high-one_lead_v_low+one_lead_v_step)/one_lead_v_step)): one_lead_v_to_idx_dic[list(range(one_lead_v_low, one_lead_v_high+ one_lead_v_step, one_lead_v_step))[i]] = i
for i in range(int((one_lead_r_high-one_lead_r_low+one_lead_r_step)/one_lead_r_step)): one_lead_r_to_idx_dic[list(range(one_lead_r_low, one_lead_r_high+one_lead_r_step, one_lead_r_step))[i]] = i
for i in range(int((one_lead_rr_high-one_lead_rr_low+one_lead_rr_step)/one_lead_rr_step)): one_lead_rr_to_idx_dic[list(range(one_lead_rr_low, one_lead_rr_high+one_lead_rr_step, one_lead_rr_step))[i]] = i

# NDD Double lane change
lc_v_low, lc_v_high, lc_v_num, lc_v_to_idx_dic = 20, 40, 21, bidict()
lc_rf_low, lc_rf_high, lc_rf_num, lc_rf_to_idx_dic = 0, 115, 116, bidict()
lc_rrf_low, lc_rrf_high, lc_rrf_num, lc_rrf_to_idx_dic = -10, 8, 19, bidict()
lc_re_low, lc_re_high, lc_re_num, lc_re_to_idx_dic = 0, 115, 116, bidict()
lc_rre_low, lc_rre_high, lc_rre_num, lc_rre_to_idx_dic = -10, 8, 19, bidict()
for i in range(lc_v_num): lc_v_to_idx_dic[list(np.linspace(lc_v_low,lc_v_high,num=lc_v_num))[i]] = i
for i in range(lc_rf_num): lc_rf_to_idx_dic[list(np.linspace(lc_rf_low,lc_rf_high,num=lc_rf_num))[i]] = i
for i in range(lc_re_num): lc_re_to_idx_dic[list(np.linspace(lc_re_low,lc_re_high,num=lc_re_num))[i]] = i
for i in range(lc_rrf_num):lc_rrf_to_idx_dic[list(np.linspace(lc_rrf_low,lc_rrf_high,num=lc_rrf_num))[i]] = i
for i in range(lc_rre_num):lc_rre_to_idx_dic[list(np.linspace(lc_rre_low,lc_rre_high,num=lc_rre_num))[i]] = i



# =========== Other parameters ============
Network_type = "Dueling"  # Normal Dueling
train_num = 0
effective_test_item = 0
total_train_num_for_epsilon = 2e5  # 1e6
Min_Mem_For_Train = 5000  # 32 5000
validation_num = 500 # 200
validation_freq = 10000 # 10000 
save_memory_freq = 30000 # 30000 
min_alpha, max_alpha, fully_compensate_train_num = 0.5, 1, 1e6
# =========== Restore training parameters =======
cav_restore_flag = False
bv_restore_flag = False
cav_restore_training_num = 0
bv_restore_training_num = 0
bv_restore_file_folder = "BV_DQN_RESULT_FIX_INI_ONE_VEH"
change_lr_flag = False
# =========== Accelerate training parameters =======
Acc_training_flag = False

# =========== Save address ==============

save_folder_name = "CAV_Dueling_SG_1400_1600_1e-5_256_64"
bv_save_folder_name = "BV_DQN_RESULT_FIX_INI_SOFTMAX_Q"
cav_training = False
bv_training = False

if cav_training:
    # Check existence of the save file address
    main_folder = os.path.abspath(".") + "/models/CAV/" + save_folder_name
    save_folder_dev = os.path.abspath(".") + "/models/CAV/" + save_folder_name + "/dev/"
    save_folder_target_net = os.path.abspath(".") + "/models/CAV/" + save_folder_name + "/target_net/"
    save_folder_memory = os.path.abspath(".") + "/models/CAV/" + save_folder_name + "/memory/"
    if os.path.exists(main_folder):pass
    else: os.mkdir(main_folder)
    if os.path.exists(save_folder_dev): pass
    else: os.mkdir(save_folder_dev)
    if os.path.exists(save_folder_target_net): pass
    else: os.mkdir(save_folder_target_net)
    if os.path.exists(save_folder_memory): pass
    else: os.mkdir(save_folder_memory)

if bv_training:
    bv_main_folder = os.path.abspath(".") + "/models/BV/" + bv_save_folder_name
    bv_save_folder_dev = os.path.abspath(".") + "/models/BV/" + bv_save_folder_name + "/dev/"
    bv_save_folder_target_net = os.path.abspath(".") + "/models/BV/" + bv_save_folder_name + "/target_net/"
    bv_save_folder_memory = os.path.abspath(".") + "/models/BV/" + bv_save_folder_name + "/memory/"
    if os.path.exists(bv_main_folder):pass
    else: os.mkdir(bv_main_folder)
    if os.path.exists(bv_save_folder_dev): pass
    else: os.mkdir(bv_save_folder_dev)
    if os.path.exists(bv_save_folder_target_net): pass
    else: os.mkdir(bv_save_folder_target_net)
    if os.path.exists(bv_save_folder_memory): pass
    else: os.mkdir(bv_save_folder_memory)

# =========== BV related ================
BV_ACTIONS = {0: 'LANE_LEFT',
            1: 'LANE_RIGHT'}
num_acc = int(((acc_high - acc_low)/acc_step) + 1)
num_non_acc = len(BV_ACTIONS)
for i in range(num_acc):
    acc = acc_to_idx_dic.inverse[i]
    BV_ACTIONS[i+num_non_acc] = str(acc)  

# =========== CAV related ================
# ACTIONS = {0: 'LANE_LEFT',
#                1: 'LANE_RIGHT',
#                2: '-4',
#                3: '-3',
#                4: '-2',
#                5: '-1',
#                6: '0',
#                7: '1',
#                8: '2'}
ACTIONS = {0: 'LANE_LEFT',
            1: 'LANE_RIGHT'}
num_acc = int(((acc_high - acc_low)/acc_step) + 1)
num_non_acc = len(ACTIONS)
for i in range(num_acc):
    acc = acc_to_idx_dic.inverse[i]
    ACTIONS[i+num_non_acc] = str(acc)  


CAV_acc_low, CAV_acc_high, CAV_acc_step = -4, 2, 0.2
num_CAV_acc = int((CAV_acc_high - CAV_acc_low)/CAV_acc_step + 1)
CAV_acc_to_idx_dic = bidict()
for i in range(num_CAV_acc): CAV_acc_to_idx_dic[list(np.arange(CAV_acc_low, CAV_acc_high + CAV_acc_step, CAV_acc_step))[i]] = i

# =========== NDD ENV para ============
safety_guard_enabled_flag, safety_guard_enabled_flag_IDM = False, True
Initial_range_adjustment_SG = 0  # m
Initial_range_adjustment_AT = 0  # m           
Stochastic_IDM_threshold = 1e-10

# =========== Test parameters ==============
r_threshold_NDD = 0  # If r < this threshold, then change to IDM vehicle
longi_safety_buffer, lateral_safety_buffer = 2, 2  # The safety buffer used to longitudinal and lateral safety guard

dis_with_CAV_in_critical_initialization = 300
critical_ini_start = initial_CAV_position - dis_with_CAV_in_critical_initialization # 0 ; initial_CAV_position - dis_with_CAV_in_critical_initialization
critical_ini_end = initial_CAV_position  # HIGHWAY_LENGTH; initial_CAV_position

LANE_CHANGE_SCALING_FACTOR = 1.
enable_One_lead_LC = True
enable_Single_LC = True
enable_Double_LC = True
enable_Cut_in_LC = True
Cut_in_veh_adj_rear_threshold = 5
ignore_adj_veh_prob, min_r_ignore = 1e-2, 5 # Probability of the ignoring the vehicle in the adjacent lane
ignore_lane_conflict_prob = 1e-4 # Probability of ignorin the lane conflict for the CAV
LC_range_threshold = 200  # [m] Now this function is void! If the range of the rear vehicle in the adjacent lane is larger than this threshold, then we consider this adj vehicle has no influence on ego-vehicle

# ============= Decompose data ================
CF_state_value = scipy.io.loadmat(os.path.abspath(".") + "/Data/Decompose/CF/" + "dangerous_state_table.mat")["dangerous_state_table"]
CF_challenge_value = scipy.io.loadmat(os.path.abspath(".") + "/Data/Decompose/CF/" + "Q_table_little.mat")["Q_table_little"]

episode = 0  # The episode number of the simulation

# CAV surrogate model IDM parameter
SM_IDM_COMFORT_ACC_MAX = 2.0  # [m/s2]  2
SM_IDM_COMFORT_ACC_MIN = -4.0  # [m/s2]  -4
SM_IDM_DISTANCE_WANTED = 5.0  # [m]  5
SM_IDM_TIME_WANTED = 1.5  # [s]  1.5
SM_IDM_DESIRED_VELOCITY = 35 # [m/s]
SM_IDM_DELTA = 4.0  # []