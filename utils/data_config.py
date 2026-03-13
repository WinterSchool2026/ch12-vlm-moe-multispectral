"""
BigEarthInfo class which contains the used band names
and band statstics that will be used to preprocess
the dataset. It also contains the class names and
matching of 43 labels to 19 labels akin to the original
Sentinel0-2 BigEarthNet dataset.

Author: Mohanad Albughdadi
Created: 2025-09-12
"""

import torch


class BigEarthNetInfo:
    BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]

    STATISTICS = {
        "min": torch.tensor([0.0] * 7),
        "max": torch.tensor(
            [63979.918, 65330.308, 65454.0, 65454.0, 65454.0, 65454.004, 65454.004]
        ),
        "mean": torch.tensor(
            [9405.194, 9649.677, 10425.686, 10444.589, 16067.627, 12699.184, 10596.475]
        ),
        "std": torch.tensor(
            [6016.344, 6095.472, 5839.462, 6068.493, 6271.440, 4256.046, 3130.776]
        ),
    }
    CLASSES = {
        19: [
            "Urban fabric",
            "Industrial or commercial units",
            "Arable land",
            "Permanent crops",
            "Pastures",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of"
            " natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland and sparsely vegetated areas",
            "Moors, heathland and sclerophyllous vegetation",
            "Transitional woodland, shrub",
            "Beaches, dunes, sands",
            "Inland wetlands",
            "Coastal wetlands",
            "Inland waters",
            "Marine waters",
        ],
        43: [
            "Continuous urban fabric",
            "Discontinuous urban fabric",
            "Industrial or commercial units",
            "Road and rail networks and associated land",
            "Port areas",
            "Airports",
            "Mineral extraction sites",
            "Dump sites",
            "Construction sites",
            "Green urban areas",
            "Sport and leisure facilities",
            "Non-irrigated arable land",
            "Permanently irrigated land",
            "Rice fields",
            "Vineyards",
            "Fruit trees and berry plantations",
            "Olive groves",
            "Pastures",
            "Annual crops associated with permanent crops",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of"
            " natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland",
            "Moors and heathland",
            "Sclerophyllous vegetation",
            "Transitional woodland/shrub",
            "Beaches, dunes, sands",
            "Bare rock",
            "Sparsely vegetated areas",
            "Burnt areas",
            "Inland marshes",
            "Peatbogs",
            "Salt marshes",
            "Salines",
            "Intertidal flats",
            "Water courses",
            "Water bodies",
            "Coastal lagoons",
            "Estuaries",
            "Sea and ocean",
        ],
    }
    MATCH_LABELS = {
        0: 0,
        1: 0,
        2: 1,
        11: 2,
        12: 2,
        13: 2,
        14: 3,
        15: 3,
        16: 3,
        18: 3,
        17: 4,
        19: 5,
        20: 6,
        21: 7,
        22: 8,
        23: 9,
        24: 10,
        25: 11,
        31: 11,
        26: 12,
        27: 12,
        28: 13,
        29: 14,
        33: 15,
        34: 15,
        35: 16,
        36: 16,
        38: 17,
        39: 17,
        40: 18,
        41: 18,
        42: 18,
    }
