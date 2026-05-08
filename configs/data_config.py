ID_COL = "PADMNO"
STEP_COL = "drug.sequence"
TIME_COL = "drug.time"
MEDICATION_SOURCE_COL = "drug.path"
TIME_LABEL_COL = "los"
DIAGNOSIS_COL = "out_diagnosis_code"
PROCEDURE_COL = "operation_NO"
SURGERY_COL = "surgery_NO"
OUTCOME_COL = TIME_LABEL_COL

STEP_INDEX_COL = "step_idx"
STEP_DURATION_COL = "step_duration"
TARGET_MED_COL = "target_med_codes"
PREV_MED_COL = "prev_med_codes"
ADDED_MED_COL = "added_med_codes"
REMOVED_MED_COL = "removed_med_codes"
RAW_REGIMEN_COL = "step_regimen_raw"

DEMOGRAPHIC_COLS = [
    "gender","age_group","marital_status","job","nationality","pat_type",
    "pay_type","pat_source","hometown","in_year","seasonality","cur_dep",
    "if_trans","admission_age","in_rank"
]

PHYSICAL_EXAM_COLS = [
    "PE_subcutaneous.bleeding","PE_expression","PE_face","PE_body.temperature",
    "PE_pulse","PE_breath","PE_SBP","PE_DBP","PE_nutrition","PE_cooperation",
    "PE_consciousness","PE_gait","PE_body.position"
]

READMISSION_COLS = [
    "main_MDD_readmission","main_psychiatry_readmission","MDD_readmission",
    "psychiatry_readmission"
]

DISEASE_COLS = [
    "out_diagnosis_NO","out_diagnosis_MDDIndex","severity","first_episode"
]

COMORBIDITY_COLS = [
    "cancer_comorbidity_NO","respiratory_comorbidity_NO","circulatory_comorbidity_NO",
    "digestive_comorbidity_NO","nervous_comorbidity_NO","endocrine_comorbidity_NO",
    "psychiatric_comorbidity_NO","comorbidity","psychiatric_comorbidity",
    "endocrine_comorbidity","nervous_comorbidity","digestive_comorbidity",
    "circulatory_comorbidity","respiratory_comorbidity","cancer_comorbidity"
]

HISTORY_COLS = [
    "history_allergy","history_blood_transfusion","history_drug_use",
    "history_surgery","history_smoking","history_alcoholism"
]

SYMPTOM_COLS = [
    "mood","bad_sleep","loss_interest","flustered","worry","tension","upset",
    "headache","dizziness","physical_discomfort","fatigue","suicide","self_harm",
    "hallucinaion","less_activity","chest_tightness","afraid","irritability",
    "fidget","slow_response","relapse","symp_worsen","core_symp","psy_symp",
    "phy_symp"
]

LAB_VALUE_COLS = [
    "RBC_CV_value","RBC_SD_value","WBCC_value","POM_value","RBCC_value.x",
    "AOM_value","RBCC_value.y","hematocrit_value","POL_value","AOL_value",
    "ARBC_HGB_value","ARBC_HGB_con_value","ARBCV_value","POB_value","POE_value",
    "hemoglobin_value","PC_value.x","PON_value","AON_value","ALA_value",
    "ASA_value","CK_value","LD_value","urea_value","TB_value","DB_value",
    "IDB_value","TP_value","albumin_value","creatinine_value","glucose_value",
    "AP_value","GT_value","sodium_value","potassium_value","chlorine_value",
    "globulin_value","WBR_value","UA_value","HD_value","cholessterol_value",
    "HDL_value","LDL_value","cystatinC_value","calcium_value","magnesium_value",
    "SIP_value","CO2CP_value","AG_value"
]

LAB_LEVEL_COLS = [
    "RBC_CV_level","RBC_SD_level","WBCC_level","POM_level","RBCC_level.x",
    "AOM_level","RBCC_level.y","hematocrit_level","POL_level","AOL_level",
    "ARBC_HGB_level","ARBC_HGB_con_level","ARBCV_level","POB_level","POE_level",
    "hemoglobin_level","PC_level.x","PON_level","AON_level","ALA_level",
    "ASA_level","CK_level","LD_level","urea_level","TB_level","DB_level",
    "IDB_level","TP_level","albumin_level","creatinine_level","glucose_level",
    "AP_level","GT_level","sodium_level","potassium_level","chlorine_level",
    "globulin_level","WBR_level","UA_level","HD_level","cholessterol_level",
    "HDL_level","LDL_level","cystatinC_level","calcium_level","magnesium_level",
    "SIP_level","CO2CP_level","AG_level"
]

BINARY_OR_CATEGORICAL_COLS = (
    DEMOGRAPHIC_COLS + PHYSICAL_EXAM_COLS + READMISSION_COLS + DISEASE_COLS +
    COMORBIDITY_COLS + HISTORY_COLS + SYMPTOM_COLS + LAB_LEVEL_COLS + [SURGERY_COL]
)

NUMERIC_COLS = ["admission_age", STEP_DURATION_COL, OUTCOME_COL] + LAB_VALUE_COLS

MIN_SUPPORT_DIAG = 5
MIN_SUPPORT_PROC = 5
MIN_SUPPORT_MED = 3
MIN_REGIMEN_SUPPORT = 3
RANDOM_STATE = 42
