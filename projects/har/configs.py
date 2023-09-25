import os

from roug_ml.configs.my_paths import data_path

# path to M Health dataset: accelerometers
M_HEALTH_PATH = os.path.join(data_path, 'MHEALTHDATASET')

# Labels
M_HEALTH_ACTIVITIES_LABELS = {0: "L0: nothing",
                              1: "L1: Standing still (1 min)",
                              2: "L2: Sitting and relaxing (1 min)",
                              3: "L3: Lying down (1 min)",
                              4: "L4: Walking (1 min)",
                              5: "L5: Climbing stairs (1 min)",
                              6: "L6: Waist bends forward (20x)",
                              7: "L7: Frontal elevation of arms (20x)",
                              8: "L8: Knees bending (crouching) (20x)",
                              9: "L9: Cycling (1 min)",
                              10: "L10: Jogging (1 min)",
                              11: "L11: Running (1 min)",
                              12: "L12: Jump front & back (20x)"}
