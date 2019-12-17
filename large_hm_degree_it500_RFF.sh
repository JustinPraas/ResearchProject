#!/usr/bin/env python3
from main import centralities, generateHeatmaps, single_combs
generateHeatmaps(["degree"], [("degree")], [0.01, 0.05, 0.1, 0.15], 500, [50, 100, 200, 400], [80, 40, 20, 10])