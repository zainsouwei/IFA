
# Confounders 
# confounds detailed in https://www.sciencedirect.com/science/article/pii/S1053811920300914 & https://www.humanconnectome.org/storage/app/media/documentation/s500/HCP500_MegaTrawl_April2015.pdf
# In Data Table: Age (Age_in_Yrs), Sex (Gender), Ethnicity (Ethnicity), Weight (Weight), Brain Size (FS_BrainSeg_Vol), Intracranial Volume (FS_IntraCranial_Vol), Confounds Modelling Slow Drift (TestRetestInterval), reconstruction code version (fMRI_3T_ReconVrs) or Acquisition Quarter (Acquisition)
# In pathfile: Head Motion (a summation over all timepoints of timepoint-to-timepoint relative head motion or average) Movement_RelativeRMS_mean.txt (Since LR RL and session scans are concateanted, take average of this average)
# Mentioned in papers but not found: variables (x, y, z, table) related to bed position in scanner
confounders = ["Age_in_Yrs", "Gender", "Ethnicity", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "fMRI_3T_ReconVrs"]