import json


class Diamond:
    """Class to store measured nitrogen aggregation information.

    The diamond is assumed to have two zones (core and rim), grown at different
    times in the diamond's growth history with distinct nitrogen
    concentrations and aggregation states.
    """
    def __init__(self,
                 age_core: int = 3520,
                 age_rim: int = 1860,
                 age_kimberlite: int = 0,
                 c_NT: int = 625, c_agg: float = 0.863,
                 r_NT: int = 801, r_agg: float = 0.197
                 ):
        """Initialise Diamond object.

        PARAMS:
        --------
        age_core | int: age of the core (Ma)
        age_rim | int: age of the rim (Ma)
        age_kimberlite | int: eruption age of the kimberlilte (Ma)
        c_NT | int: total nitrogen concentration in the core (ppm)
        c_agg | float: aggregation state of the core (proportion of N in B)
        r_NT | int: total nitrogen concentration in the rim (ppm)
        r_agg | float: aggregation state of the rim (proportion of N in B)
        """

        self.age_core = age_core
        self.age_rim = age_rim
        self.age_kimberlite = age_kimberlite
        self.c_NT = c_NT
        self.c_agg = c_agg
        self.r_NT = r_NT
        self.r_agg = r_agg

    @classmethod
    def from_json(cls, filepath: str):
        """Create Diamond instance from JSON file.

        PARAMS:
        -------
        filepath | str : path to JSON file

        RETURNS:
        --------
        Diamond instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls(
            age_core=data['age_core'],
            age_rim=data['age_rim'],
            age_kimberlite=data['age_kimberlite'],
            c_NT=data['c_NT'],
            c_agg=data['c_agg'],
            r_NT=data['r_NT'],
            r_agg=data['r_agg']
        )

    def to_json(self, filepath: str):
        """Save Diamond instance to JSON file.

        PARAMS:
        -------
        filepath | str : path to JSON file
        """
        data = {
            'age_core': self.age_core,
            'age_rim': self.age_rim,
            'age_kimberlite': self.age_kimberlite,
            'c_NT': self.c_NT,
            'c_agg': self.c_agg,
            'r_NT': self.r_NT,
            'r_agg': self.r_agg
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def __str__(self):
        """Return human-readable string represantation.
        """

        return (
            "Diamond with\n"
            f"- core age {self.age_core} Ma,\n"
            f"- rim age {self.age_rim} Ma,\n"
            f"- kimberlite age {self.age_kimberlite} Ma.\n"
            f"- Core: [N_T] {self.c_NT} ppm, {self.c_agg*100}%B.\n"
            f"- Rim: [N_T] {self.r_NT} ppm, {self.r_agg*100}%B."
        )
