"""
execute_GA2.pyのパラメータを探索するプログラム(自作)

仕様(仮):
    1. pdbとfastaと乱数シード値は固定する
    2. 個体数は計算機のスペックと相談するので手動で単体コードを動かして調整
    3. 世代数は各世代の平均適応度と世代間の適応度上昇の加速度をみてプログラム側が毎回決定→グラフ表示?
    4. 変異と組み換えをfor文で回して探索
    5. 選択方式(エリートの選択数も含む)は余裕があれば関数のバリエーション作って変えてみる?
    6. 結果はグラフを画像にして保存する & csvにしてログに残すとかかな?
        xyzの角度と適応度と世代数(到達時間として近似)くらいあればよいはず
        
    7. なお余力があればクォータニオンverを作ってみてもいい
    
"""

import numpy as np
import matplotlib.pyplot as plt
from execute_GAv2 import GeneticAlgorithmAligner as GAv2

LOG_FILE = "GAlog.txt"

POPULATION_NUM = 150
GENERATION_MAX = 200
MUT_MIN, MUT_MAX, MUT_STEP = 0.1, 1, 0.1
REC_MIN, REC_MAX, REC_STEP = 0.1, 1, 0.1

GAmodels = [[GAv2(POPULATION_NUM,GENERATION_MAX,mut_rate,rec_rate) for mut_rate in np.arange(MUT_MIN, MUT_MAX, MUT_STEP)] for rec_rate in np.arange(REC_MIN, REC_MAX, REC_STEP)]

running = np.ones_like(GAmodels).astype(bool)
groups = np.array([[ga.prepare_ga() for ga in _] for _ in GAmodels])
temp = np.zeros_like(groups)
fitnesses = np.zeros_like(groups)

for gen in range(GENERATION_MAX):
    temp = np.array([[
            GAmodels[idx,idy].recombination(GAmodels[idx,idy].mutation(groups[idx, idy]))
            for idy in len(GAmodels[0])] for idx in len(GAmodels)]
            )
    fitnesses = np.array([[[GAmodels[idx,idy].calc_fitness(pop) for pop in temp[idx,idy]] for idy in range(len(GAmodels[0]))]for idx in range(len(GAmodels))])
    groups = np.array([[GAmodels[idx, idy].selection(temp[idx,idy], fitnesses[idx,idy]) for idy in range(len(GAmodels[0]))] for idx in range(len(GAmodels))])
    
