# bacteria_p.py
import random
import numpy as np
from parameters_5 import (
    INITIAL_ENERGY, CHANNEL_COST, DIVISION_THRESHOLD,
    BASE_METABOLIC_COST, MAINTENANCE_COST,
    UPTAKE_PER_CHANNEL, SUPPRESSION_K, BUILD_PROB_SLOPE,
    INITIAL_NUTRIENT, DEATH_PROB, DIVISION_SWITCH_PROB,
    DIVISION_BIAS_MEAN, DIVISION_BIAS_SD,
    DIVISION_BIAS_MIN, DIVISION_BIAS_MAX
)

class Bacterium:
    """
    Each bacterium keeps independent channel counts for A, B and C nutrients.
    ``type`` determines which nutrient it can *build* channels for.
    """
    def __init__(
        self,
        energy=INITIAL_ENERGY,
        cell_id=0,
        channels_A=0,
        channels_B=0,
        channels_C=0,
        btype='A'          # 'A', 'B' or 'C'
    ):
        self.energy       = energy
        self.channels_A   = channels_A
        self.channels_B   = channels_B
        self.channels_C   = channels_C
        self.type         = btype
        self.id           = cell_id
        self.dead         = False

        # cached constants
        self.channel_cost        = CHANNEL_COST
        self.maintenance_cost    = MAINTENANCE_COST
        self.base_metabolic_cost = BASE_METABOLIC_COST
        self.div_threshold       = DIVISION_THRESHOLD

    # ─────────────────────────── helpers ────────────────────────────
    @staticmethod
    def _build_prob(local_nutrient, total_channels):
        nutr_drive  = BUILD_PROB_SLOPE * local_nutrient
        suppression = total_channels / (total_channels + SUPPRESSION_K)
        return max(0.0, min(1.0, nutr_drive * (1.0 - suppression)))

    # ─────────────────────────── main update ────────────────────────
    def consume(self, local_A, local_B, local_C):
        """One timestep of metabolism, uptake, possible channel-building."""
        # (0) random death
        if random.random() < DEATH_PROB:
            self.dead = True
            return

        # (1) uptake
        upA = min(local_A, self.channels_A * UPTAKE_PER_CHANNEL)
        upB = min(local_B, self.channels_B * UPTAKE_PER_CHANNEL)
        upC = min(local_C, self.channels_C * UPTAKE_PER_CHANNEL)
        self.energy += (upA + upB + upC)

        # (2) costs
        tot_ch = self.channels_A + self.channels_B + self.channels_C
        self.energy -= (
            self.base_metabolic_cost
            + self.maintenance_cost * tot_ch
            + self.channel_cost    * tot_ch
        )

        # (3) potential channel build
        if self.energy > 0.2 * self.div_threshold:
            if self.type == 'A' and local_A > 0:
                if random.random() < self._build_prob(local_A, tot_ch) * min(local_A/INITIAL_NUTRIENT,1.0):
                    self.channels_A += 1
            elif self.type == 'B' and local_B > 0:
                if random.random() < self._build_prob(local_B, tot_ch) * min(local_B/INITIAL_NUTRIENT,1.0):
                    self.channels_B += 1
            elif self.type == 'C' and local_C > 0:
                if random.random() < self._build_prob(local_C, tot_ch) * min(local_C/INITIAL_NUTRIENT,1.0):
                    self.channels_C += 1

        # keep everything non-negative
        self.energy     = max(self.energy, 0.0)
        self.channels_A = max(self.channels_A, 0)
        self.channels_B = max(self.channels_B, 0)
        self.channels_C = max(self.channels_C, 0)

    # ─────────────────────────── division logic ─────────────────────
    def ready_to_divide(self):
        tot_ch = self.channels_A + self.channels_B + self.channels_C
        return (not self.dead) and self.energy >= self.div_threshold and tot_ch >= 2

    def divide(self, new_id):
        if self.dead:
            return None

        r = np.clip(
            np.random.normal(DIVISION_BIAS_MEAN, DIVISION_BIAS_SD),
            DIVISION_BIAS_MIN, DIVISION_BIAS_MAX
        )

        # split energy
        child_energy  = self.energy * (1.0 - r)
        self.energy  *= r

        # deterministic split of channel integers
        def split(count):
            parent = int(round(count * r))
            return parent, count - parent

        self.channels_A, child_A = split(self.channels_A)
        self.channels_B, child_B = split(self.channels_B)
        self.channels_C, child_C = split(self.channels_C)

        # daughter type : keep / switch among the other two
        if random.random() < DIVISION_SWITCH_PROB:
            other = {'A':['B','C'],'B':['A','C'],'C':['A','B']}[self.type]
            new_type = random.choice(other)
        else:
            new_type = self.type

        return Bacterium(
            energy=child_energy, cell_id=new_id,
            channels_A=child_A, channels_B=child_B, channels_C=child_C,
            btype=new_type
        )
