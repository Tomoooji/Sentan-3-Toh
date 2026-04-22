import os
import random
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. ファイル読み書きライブラリ
# =============================================================================
def read_fasta(file_path):
    seqs = []
    curr_seq = ""
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if curr_seq:
                    seqs.append(curr_seq)
                    curr_seq = ""
            else:
                curr_seq += line
    if curr_seq:
        seqs.append(curr_seq)
    return seqs

def read_pdb(file_path):
    atoms = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                atom = {
                    'line': line,
                    'elety': line[12:16].strip(),
                    'resid': line[17:20].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54])
                }
                atoms.append(atom)
    return atoms

def write_pdb(atoms, out_path):
    with open(out_path, "w") as f:
        for a in atoms:
            line = a['line']
            new_line = f"{line[:30]}{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}{line[54:]}"
            f.write(new_line)

# =============================================================================
# 2. 数学・行列演算関数
# =============================================================================
# X/Y/Z軸回りにtheta回転させる行列(回転行列)←線形代数学で一瞬だけやったはず
def rx(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

# =============================================================================
# 3. 遺伝的アルゴリズムクラス
# =============================================================================
class GeneticAlgorithmAligner:
    def __init__(self, pdb1, pdb2, aln, pop_size, gen_num, mut_rate, rec_rate):
        # 主要な変数の初期化
        self.pdb1 = pdb1
        self.pdb2 = pdb2
        self.aln = aln
        self.pop_size = pop_size
        self.gen_num = gen_num
        self.mut_rate = mut_rate
        self.rec_rate = rec_rate
        self.redx = 1.0
        
        # 出力ファイル名の自動生成
        base1 = os.path.basename(self.pdb1).split('.')[0]
        base2 = os.path.basename(self.pdb2).split('.')[0]
        suffix = f"_{self.pop_size}_{self.gen_num}_{self.mut_rate}_{self.rec_rate}_{self.redx}"
        self.out_pdb1 = f"{base1}{suffix}.pdb"
        self.out_pdb2 = f"{base2}{suffix}.pdb"
        self.out_plot = f"plot_{base1}_{base2}{suffix}.png"

        # 主要な変数の初期化
        self.count = 0
        self.rt = 1.0
        self.recd = np.zeros(self.gen_num)
        self.pos = None
        self.ca1_coords = None
        self.ca2_coords = None
        self.p1x = None
        self.p2x = None
        self.align_size = 0

    def prepare_ga(self):
        # 出力先ファイル名の重複防止
        if os.path.exists(self.out_pdb1) or os.path.exists(self.out_pdb2) or os.path.exists(self.out_plot):
            raise FileExistsError(f"エラー：出力ファイル ({self.out_plot} 等) が既に存在します。古いファイルを削除するかリネームしてください。")

        # 入力ファイル読み込み
        print("-----> two pdb files and a fasta alignment file are read")
        seqs = read_fasta(self.aln)# -> 11行目~
        p1_atoms = read_pdb(self.pdb1)# -> 27行目~
        p2_atoms = read_pdb(self.pdb2)# -> 27行目~

        print("-----> correspondece between alignment and pdb is made")
        self.align_size = len(seqs[0])
        self.pos = np.zeros((2, self.align_size), dtype=int) - 1

        for i in range(2):
            site = 0
            for j in range(self.align_size):
                if seqs[i][j] != '-':
                    self.pos[i, j] = site
                    site += 1
        
        # Cα原子の座標を行列として抽出
        print("-----> CA coordinates are obtained from PDB data")
        # atomのeletyがCAのインデックスの抽出
        ca1_indices = [i for i, a in enumerate(p1_atoms) if a['elety'] == 'CA']
        ca2_indices = [i for i, a in enumerate(p2_atoms) if a['elety'] == 'CA']

        # 抽出したインデックスをもとにx,y,z座標を格納
        ca1_raw = np.array([[p1_atoms[i]['x'], p1_atoms[i]['y'], p1_atoms[i]['z']] for i in ca1_indices])
        ca2_raw = np.array([[p2_atoms[i]['x'], p2_atoms[i]['y'], p2_atoms[i]['z']] for i in ca2_indices])

        # 格納した座標から重心を計算
        print("-----> Calculation of geometric center of 1st and 2nd PDB file")
        ca1_center = np.mean(ca1_raw, axis=0)
        ca2_center = np.mean(ca2_raw, axis=0)

        print("-----> The geometric centers of the CA coordinates are set to the origin")
        # 計算用に"深いコピー"を作成
        self.p1x = copy.deepcopy(p1_atoms)
        self.p2x = copy.deepcopy(p2_atoms)
        # 計算した重心を引き算して相対位置に変換
        for a in self.p1x:
            a['x'] -= ca1_center[0]
            a['y'] -= ca1_center[1]
            a['z'] -= ca1_center[2]
        # 計算した重心を引き算して相対位置に変換
        for a in self.p2x:
            a['x'] -= ca2_center[0]
            a['y'] -= ca2_center[1]
            a['z'] -= ca2_center[2]

        # Numpy配列に変換
        self.ca1_coords = np.array([[self.p1x[i]['x'], self.p1x[i]['y'], self.p1x[i]['z']] for i in ca1_indices])
        self.ca2_coords = np.array([[self.p2x[i]['x'], self.p2x[i]['y'], self.p2x[i]['z']] for i in ca2_indices])

        # 初期集団の作成
        print("-----> Population is generated")
        population = np.random.rand(self.pop_size, 3) * 2 * np.pi
        # RMSDの初期値を計算
        initial_rmsd = (1 / self.calc_fitness(population[0])) - 0.01
        print(f"initial rmsd = {initial_rmsd:.4f}\n")
        
        return population # 戻り値として初期化した集団を返す

    # 適応度を計算する
    def calc_fitness(self, angles):
        # X,Y,Z軸回りの回転行列を順番にかけておく (@は行列の掛け算,mtxはmatrix)
        rmtx = rx(angles[0]) @ ry(angles[1]) @ rz(angles[2])
        rmsd = 0.0
        nst = 0

        # 各点の座標に回転行列をかけてRMSDとNSTを計算
        for i in range(self.align_size):
            p1_idx = self.pos[0, i]
            p2_idx = self.pos[1, i]
            if p1_idx != -1 and p2_idx != -1:
                c1 = self.ca1_coords[p1_idx]
                c2 = self.ca2_coords[p2_idx]
                cc = rmtx @ c1
                rmsd += np.sum((cc - c2)**2)
                nst += 1

        rmsd = np.sqrt(rmsd / nst)
        return 1.0 / (rmsd + 0.01) #RMSDを返す

    # 引数rndで与えられた乱数値をもとに角度thetaを増減して返す
    def mod_angle(self, rnd, theta):
        # self.rt(初期値:1)が変異の大きさを制御している
        delta = 2 * np.pi * random.random() * self.rt
        # 乱数が変異を起こす確率(閾値)より大きければ変異は起きない
        if rnd > self.mut_rate:
            return theta
        else:
            # 乱数が閾値以下のとき50%の確率でdelta分thetaが増える
            if random.random() > 0.5:
                x = theta + delta
                if x > 2 * np.pi: x -= 2 * np.pi
            #                 --50%の確率でdelta分thetaが減る
            else:
                x = theta - delta
                if x < 0: x += 2 * np.pi
            return x

    # 与えられた集団中の各個体に対してX,Y,Zの回転角に変異率に応じて突然変異を起こさせる
    def mutation(self, population):
        pop_size = len(population)
        mutants = []
        for i in range(pop_size):
            mx, my, mz = random.random(), random.random(), random.random()
            if mx > self.mut_rate and my > self.mut_rate and mz > self.mut_rate:
                continue
            new_x = self.mod_angle(mx, population[i, 0])
            new_y = self.mod_angle(my, population[i, 1])
            new_z = self.mod_angle(mz, population[i, 2])
            mutants.append([new_x, new_y, new_z])
            
        # 変異した個体だけmutantsに格納して個体数を出力し、元の集団と合流させる
        print(f"No of mutants = {len(mutants)}")
        if mutants:
            return np.vstack([population, np.array(mutants)])
        return population # またしても集団を返す

    # 与えられた集団内で組み換えを行う
    def recombination(self, population):
        # 0~1の乱数配列を作って組み換え率(閾値)self.rec_rate未満だった要素の数だけ組み換えを起こす
        print("-----> Recombination: ", end="")
        pop_size = len(population)
        rsize = np.sum(np.random.rand(pop_size) < self.rec_rate)
        print(f"rsize = {rsize}")
        print(f"No of Recombinants = {rsize}")
        
        recombinants = []
        for _ in range(rsize):
            # ランダムに選んだ2個体の回転角を組み替えてrecombinantsに格納
            mem1 = random.randint(0, pop_size - 1)
            mem2 = random.randint(0, pop_size - 1)
            rp = random.randint(1, 2)
            if rp == 1:
                recombinants.append([population[mem1, 0], population[mem2, 1], population[mem2, 2]])
            else:
                recombinants.append([population[mem1, 0], population[mem1, 1], population[mem2, 2]])
                
        # 組み換え後の個体群をもとの集団に戻す
        if recombinants:
            return np.vstack([population, np.array(recombinants)])
        return population # 組み換え後の集団を返す

    # 
    def selection(self, population, fitness, gen_idx):
        print("-----> Sampling Next Generation")
        ord_idx = np.argsort(fitness)[::-1]# 集団中の個体のインデックスを適応度が大きい順に取得
        sizes = len(population)

        wheel = np.zeros(sizes)
        wheel[0] = fitness[ord_idx[0]]# 適応度が最大の個体
        # 適応度が小さい順に加算しながらリストアップ?
        for j in range(1, sizes):
            wheel[j] = wheel[j-1] + fitness[ord_idx[j]]

        rnd = np.sort(np.random.rand(self.pop_size) * wheel[-1])

        mem = np.zeros(sizes, dtype=int)
        for j in range(sizes):
            if j == 0:
                mem[0] = np.sum(rnd < wheel[0])
            else:
                mem[j] = np.sum(rnd < wheel[j]) - np.sum(rnd < wheel[j-1])

        print("-----> Elite Selection")
        if mem[0] == 0:
            mem[0] = 1
            for j in range(sizes - 1, 0, -1):
                if mem[j] != 0:
                    mem[j] -= 1
                    break

        print("-----> Make Next Generation")
        # エリート以外をリストアップ
        new_population = []
        for j in range(sizes):
            if mem[j] > 0:
                for _ in range(mem[j]):
                    new_population.append(population[ord_idx[j]])

        # 各世代の最良適応度を保存しておく(逆数から0.01を引いてるのはグラフ描画用の調整?)
        self.recd[gen_idx] = (1.0 / fitness[ord_idx[0]]) - 0.01

        # 世代ごとの材料適応度が15世代変化しなかった場合に変異率を変化させる(適応的変異?)
        # 実際には適応度の変化率(self.rtとself.redx)が1に設定されてるため機能していない
        if gen_idx > 0:
            if self.recd[gen_idx] == self.recd[gen_idx - 1]:
                self.count += 1
            else:
                self.count = 0
            if self.count > 15:
                self.count = 0
                self.rt *= self.redx

        print(f"Min RMSD of Generation {gen_idx + 1} = {self.recd[gen_idx]:.5f}\n")
        return np.array(new_population)[:self.pop_size]

    # 結果出力用
    def output_results(self, final_population):
        fitness = np.array([self.calc_fitness(ind) for ind in final_population])
        best_idx = np.argmax(fitness)
        best_angles = final_population[best_idx]
        final_rmsd = (1.0 / fitness[best_idx]) - 0.01

        rmtx = rx(best_angles[0]) @ ry(best_angles[1]) @ rz(best_angles[2])
        for a in self.p1x:
            cc = rmtx @ np.array([a['x'], a['y'], a['z']])
            a['x'], a['y'], a['z'] = cc[0], cc[1], cc[2]

        write_pdb(self.p1x, self.out_pdb1)
        write_pdb(self.p2x, self.out_pdb2)

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, self.gen_num + 1), self.recd, '-')
        plt.xlabel("Generation")
        plt.ylabel("RMSD")
        plt.title(f"Filename: {self.out_plot}")
        
        info_text = (f"FinalRMSD: {final_rmsd:.4f}\n"
                     f"PopulationSize: {self.pop_size}\n"
                     f"GenerationNumber: {self.gen_num}\n"
                     f"MutationRate: {self.mut_rate}\n"
                     f"RecombinationRate: {self.rec_rate}")
        plt.gca().text(0.95, 0.95, info_text, transform=plt.gca().transAxes, 
                       fontsize=10, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.savefig(self.out_plot)
        plt.close()
        print(f"GA Process Finished Successfully.")
        print(f"Outputs generated:\n- {self.out_pdb1}\n- {self.out_pdb2}\n- {self.out_plot}")

# =============================================================================
# 4. メイン実行プロセス (コマンドライン引数パース)
# =============================================================================
if __name__ == "__main__":
    # 引数パーサーの設定
    parser = argparse.ArgumentParser(description="PDB構造アラインメントのための遺伝的アルゴリズム")
    
    # 必須パラメータ
    parser.add_argument("--pdb1", type=str, required=True, help="入力立体構造1のPDBファイル名")
    parser.add_argument("--pdb2", type=str, required=True, help="入力立体構造2のPDBファイル名")
    parser.add_argument("--aln", type=str, required=True, help="アラインメントファイル名 (FASTA形式)")
    
    # オプションパラメータ（デフォルト値あり）
    parser.add_argument("--pop_size", type=int, default=100, help="集団サイズ (デフォルト: 100)")
    parser.add_argument("--gen_num", type=int, default=100, help="世代数 (デフォルト: 100)")
    parser.add_argument("--mut_rate", type=float, default=0.7, help="変異率 (デフォルト: 0.7)")
    parser.add_argument("--rec_rate", type=float, default=0.7, help="組み換え率 (デフォルト: 0.7)")

    # コマンドライン引数をパース
    args = parser.parse_args()

    # シード値の固定（必要な場合はコメントアウトを解除）
    # random.seed(1)
    # np.random.seed(1)

    # GAクラスの初期化（パースした引数を渡す）
    ga = GeneticAlgorithmAligner(
        pdb1=args.pdb1,
        pdb2=args.pdb2,
        aln=args.aln,
        pop_size=args.pop_size,
        gen_num=args.gen_num,
        mut_rate=args.mut_rate,
        rec_rate=args.rec_rate
    )

    # 初期集団の作成
    population = ga.prepare_ga()
    
    # 世代ごとのループ
    for i in range(ga.gen_num):
        mutated = ga.mutation(population)
        recombinated = ga.recombination(mutated)
        
        print("-----> Calculation of Fitness")
        fitness_vals = np.array([ga.calc_fitness(ind) for ind in recombinated])
        
        population = ga.selection(recombinated, fitness_vals, i)

    # 結果の出力
    ga.output_results(population)

    
