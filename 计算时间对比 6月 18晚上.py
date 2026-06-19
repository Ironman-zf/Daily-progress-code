import os
import re
import time
import copy
import joblib
import warnings
import pandas as pd
import numpy as np
from types import SimpleNamespace
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from pathlib import Path
import torch
import torch.nn as nn
from scipy.stats import qmc
from compression_system1 import compression_system4

class PhysicsAwareScaler:
    """
    X 缩放器：只对连续物理量（功率、背压）归一化，绝对不碰 0/1 开关
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X):
        X_out = np.array(X, copy=True)
        X_out[:, 0:2] = self.scaler.fit_transform(X_out[:, 0:2])
        return X_out

    def transform(self, X):
        X_out = np.array(X, copy=True)
        X_out[:, 0:2] = self.scaler.transform(X_out[:, 0:2])
        return X_out

    def inverse_transform(self, X_scaled):
        X_out = np.array(X_scaled, copy=True)
        X_out[:, 0:2] = self.scaler.inverse_transform(X_out[:, 0:2])
        return X_out


class PassThroughScaler:
    """
    Y 缩放器（直通车）：转速比 [0.7, 1.1] 和效率 [0.8, 0.9] 已是完美靶值，禁止数学平移。
    """

    def fit_transform(self, Y): return np.array(Y, copy=True)

    def transform(self, Y): return np.array(Y, copy=True)

    def inverse_transform(self, Y_scaled): return np.array(Y_scaled, copy=True)


def configure_design_parameters():
    des = SimpleNamespace()
    des.epsilon = [3.2365, 3.2365, 3.2365, 3.2365]
    des.total_epsilon = 100.0
    des.eta = 0.84
    des.Gc = 10.138
    des.InT = [298.0, 303.0, 303.0, 303.0, 303.0]
    des.OutT = [439.9, 447.29, 447.29, 447.29]
    des.InP = [1.01325, 3.037, 9.628, 30.96]
    des.A = [1310.0 * 150, 1344.0 * 150, 1785.0 * 150, 2366.0 * 150]
    des.des_delta_P = 0.2
    des.n_ = [1.0, 1.0, 1.0, 1.0]
    des.T_water_in = 298.15
    des.P_water_bar = 18.0
    des.ratio = 0.25
    return des


# ==========================================
# 2. 数据解析模块
# ==========================================
def parse_physics_from_filename(filename):
    bp = float(m.group(1)) if (m := re.search(r'-(\d+)-compressor', os.path.basename(filename))) else 0.0
    acts = [int(s) for s in os.path.basename(filename).split('-' + str(int(bp)))[0].split('-') if s.isdigit()]
    return bp, 1.0 if 1 in acts else 0.0, 1.0 if 2 in acts else 0.0, 1.0 if 3 in acts else 0.0, 1.0 if 4 in acts else 0.0



def load_and_sample_unseen_data(unseen_file_paths, sample_count=50):
    valid_dataframes_list = []

    for current_file_path in unseen_file_paths:
        if not os.path.exists(current_file_path):
            continue

        # --- 1. 基础异常值清洗 ---
        raw_dataframe = pd.read_csv(current_file_path)

        if 'Exit_Flag' in raw_dataframe.columns:
            raw_dataframe = raw_dataframe[raw_dataframe['Exit_Flag'] == 1]

        clean_dataframe = raw_dataframe.dropna(how='all').dropna(axis=1, how='all').dropna(how='any').copy()

        if clean_dataframe.empty:
            continue

        # --- 2. 物理去重处理  ---
        reference_column = 'Gc_kg_s' if 'Gc_kg_s' in clean_dataframe.columns else 'Power_total_W'
        unique_dataframe = clean_dataframe.drop_duplicates(subset=[reference_column]).copy()
        sorted_dataframe = unique_dataframe.sort_values(by=reference_column).reset_index(drop=True)

        # --- 3.  物理单调性处理 ---
        if 'Gc_kg_s' in sorted_dataframe.columns:
            # 寻找总功率的绝对峰值索引
            max_power_index = sorted_dataframe['Power_total_W'].idxmax()

            # 一刀切断峰后多解恶化区 (单调性清洗完毕)
            physically_valid_dataframe = sorted_dataframe.iloc[:max_power_index + 1].copy()
        else:
            physically_valid_dataframe = sorted_dataframe.copy()

        # --- 4. 物理边界条件 ---
        ip_bar, flag_1, flag_2, flag_3, flag_4 = parse_physics_from_filename(current_file_path)

        physically_valid_dataframe['Inlet_Pressure_bar'] = ip_bar
        physically_valid_dataframe['Flag_S1_Active'] = flag_1
        physically_valid_dataframe['Flag_S2_Active'] = flag_2
        physically_valid_dataframe['Flag_S3_Active'] = flag_3
        physically_valid_dataframe['Flag_S4_Active'] = flag_4

        # 将当前文件清洗后的【全量】有效数据加入全局池，绝不在此处进行局部抽样
        valid_dataframes_list.append(physically_valid_dataframe)

    # --- 5. 跨越全域拼接全局数据池 ---
    if not valid_dataframes_list:
        print("      [警告] 未能从指定路径中提取到任何有效数据！")
        return pd.DataFrame()

    global_master_dataframe = pd.concat(valid_dataframes_list, axis=0, ignore_index=True)

    # --- 6. 终极全局随机抽样 ---
    total_valid_samples = len(global_master_dataframe)
    actual_sample_count = min(sample_count, total_valid_samples)

    # 锁定随机种子，确保基准测试 100% 可被复现
    final_sampled_dataframe = global_master_dataframe.sample(n=actual_sample_count, random_state=42)
    final_sampled_dataframe = final_sampled_dataframe.reset_index(drop=True)

    print(
        f"      [数据处理] 全局有效物理数据共 {total_valid_samples} 条，已成功随机抽取 {actual_sample_count} 条点位用于对比。")
    return final_sampled_dataframe



def thermodynamic_forward_model(n_array, mass_flow,  base_des_para):
    des_local = copy.deepcopy(base_des_para)
    des_local.n_ = np.array(n_array, dtype=float)
    ac_local = SimpleNamespace(Gc=mass_flow, T_tank_out=298.15)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = compression_system4(des_local, ac_local)

            # 🚨 修正1：遇到不可行物理区域，必须抛出 NaN 破坏平坦梯度，禁止返回常数
            if not out.get('feasible', False): return 1e10, 1e10, 0.0

            # 🚨 修正2：字典键值必须与您的真实物理导出列名严格一致！
            power = np.nansum(out.get('Power_out', [1e10]))
            pressure = out.get('P_air_comp', [])
            eff = out.get('eta_ex_system', 0.0)

            return power, pressure[-1], eff
    except Exception:
        # 捕获所有由于气动矩阵求逆或查表导致的底层错误
        return 1e10, 1e10, 0.0


def global_optimization_traditional(target_power, target_pressure, active_flags, base_des_para,
                                    init_gc=None, init_spd=None, n_restarts=1, seed=None,
                                    power_tol=0.001, bp_tol=0.001): # 💡 建议设为 0.001 (0.1%)
    rng = np.random.default_rng(seed)
    last_x, last_res = None, (0.0, 0.0, 0.0)

    def eval_phys(x):
        nonlocal last_x, last_res
        if last_x is None or not np.array_equal(x, last_x):
            last_res = thermodynamic_forward_model(x[0:4], x[4], base_des_para)
            last_x = np.copy(x)
        return last_res

    def obj(x):
        return -eval_phys(x)[2]

    # ---------------------------------------------------------
    # 🚨 修复后的容差带不等式约束
    # ---------------------------------------------------------
    # 1. 功率约束： -power_tol <= D_pow <= power_tol
    def safe_rel_err(val, target):
        if not np.isfinite(val) or not np.isfinite(target) or abs(target) < 1e-12:
            return -1e9
        return (val - target) / target

    def c_pow_upper(x):
        return power_tol - safe_rel_err(eval_phys(x)[0], target_power)

    def c_pow_lower(x):
        return safe_rel_err(eval_phys(x)[0], target_power) + power_tol

    def c_pres_upper(x):
        return bp_tol - safe_rel_err(eval_phys(x)[1], target_pressure)

    def c_pres_lower(x):
        return safe_rel_err(eval_phys(x)[1], target_pressure) + bp_tol

    bounds_spd = [(0.7, 1.1) if f == 1.0 else (1.0, 1.0) for f in active_flags]
    gc_bounds = (3, 25.0)
    full_bounds = bounds_spd + [gc_bounds]

    # 将约束类型改为不等式
    constraints = [
        {'type': 'ineq', 'fun': c_pow_upper},
        {'type': 'ineq', 'fun': c_pow_lower},
        {'type': 'ineq', 'fun': c_pres_upper},
        {'type': 'ineq', 'fun': c_pres_lower}
    ]

    best_res, best_eff = None, -np.inf

    # ==========================================
    # 🚨 核心升级：构建拉丁超立方 (LHS) 均匀先验矩阵
    # ==========================================
    # 1. 拆解并提取物理边界阵列
    lower_bounds = np.array([b[0] for b in full_bounds])
    upper_bounds = np.array([b[1] for b in full_bounds])

    # 2. 一次性生成所有所需的探索点 (留出 1 个名额给物理中点)
    n_lhs_samples = n_restarts - 1
    if n_lhs_samples > 0:
        lhs_sampler = qmc.LatinHypercube(d=len(full_bounds), seed=seed)
        lhs_standard_samples = lhs_sampler.random(n=n_lhs_samples)
        lhs_physical_samples = lower_bounds + lhs_standard_samples * (upper_bounds - lower_bounds)


    for k in range(n_restarts):
        current_x0 = []

        current_x0 = np.array([1.0, 1.0, 1.0, 1.0, 10.138])
        try:
            res = minimize(
                fun=obj, x0=current_x0, method='SLSQP', bounds=full_bounds,
                constraints=constraints, options={'ftol': 1e-4, 'maxiter': 1000, 'disp': False,'eps': 1e-5}
            )
            # 物理可行性校验
            if res.success and np.isfinite(res.fun) and (-res.fun > 0.01):
                current_efficiency = -res.fun
                if current_efficiency > best_eff:
                    best_eff = current_efficiency
                    best_res = res
        except Exception:
            continue

    if best_res is None:
        return np.array([np.nan, np.nan, np.nan, np.nan]), np.nan, np.nan, False

    return best_res.x[0:4], best_res.x[4], -best_res.fun, True


class PhysicsEmbeddedMLP(nn.Module):
    """
    🚨 压缩机专属 PINN 架构：
    从底层网络保证：当级关闭时(Flag=0)，输出转速比绝对为 1.0
    """

    def __init__(self, in_dim, out_dim, hidden_layers):
        super().__init__()
        layers = []
        curr = in_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(curr, h), nn.ReLU()])
            curr = h
        layers.append(nn.Linear(curr, out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        raw_output = self.network(x)
        flags = x[:, 2:6]
        # 核心物理方程：预测值 * 激活标志 + 1.0 * (1 - 激活标志)
        masked_speeds = raw_output[:, 0:4] * flags + 1.0 * (1.0 - flags)
        efficiency = raw_output[:, 4].unsqueeze(1)
        return torch.cat((masked_speeds, efficiency), dim=1)


class TorchMLPWrapper:
    def __init__(self, pytorch_model, compute_device):
        self.model = pytorch_model
        self.device = compute_device

    def predict(self, input_features):
        self.model.eval()
        with torch.no_grad():
            feature_tensor = torch.tensor(input_features, dtype=torch.float32).to(self.device)
            return self.model(feature_tensor).cpu().numpy()

def visualize_benchmark_3panels(ai_t, trad_t, ai_eff_err, trad_eff_err, ai_act_err, trad_act_err, act_name, act_unit, filename):
    plt.rcParams['font.weight'] = 'black'; plt.rcParams['axes.labelweight'] = 'black'; plt.rcParams['axes.titleweight'] = 'black'
    fig, (ax_time, ax_eff, ax_act) = plt.subplots(1, 3, figsize=(28, 8))
    idx = np.arange(1, len(ai_t) + 1)

    ax_time.plot(idx, trad_t, 's-', color='#d62728', lw=2, ms=8, alpha=0.8, label='Traditional (5D Opt)')
    ax_time.plot(idx, ai_t, 'o-', color='#1f77b4', lw=2, ms=8, alpha=0.8, label='AI (PyTorch MLP)')
    ax_time.set_title('Computational Time per Sample', fontsize=20, pad=15)
    ax_time.set_ylabel('Execution Time (Seconds)', fontsize=16)

    safe_trad_eff = np.where(np.isnan(trad_eff_err), np.nan, np.clip(trad_eff_err, 1e-6, None))
    safe_ai_eff = np.clip(ai_eff_err, 1e-6, None)
    ax_eff.plot(idx, safe_trad_eff, 's-', color='#d62728', lw=2, ms=8, alpha=0.8, label='Traditional (5D Opt)')
    ax_eff.plot(idx, safe_ai_eff, 'o-', color='#1f77b4', lw=2, ms=8, alpha=0.8, label='AI (PyTorch MLP)')
    ax_eff.set_title('Efficiency Error', fontsize=20, pad=15)
    ax_eff.set_ylabel('Absolute Error', fontsize=16)

    ax_act.plot(idx, trad_act_err, 's-', color='#d62728', lw=2, ms=8, alpha=0.8, label='Traditional (5D Opt)')
    ax_act.plot(idx, ai_act_err, 'o-', color='#1f77b4', lw=2, ms=8, alpha=0.8, label='AI (PyTorch MLP)')
    ax_act.set_title(f'{act_name} Prediction Error', fontsize=20, pad=15)
    ax_act.set_ylabel(f'Absolute Error ({act_unit})', fontsize=16)

    for ax in [ax_time, ax_eff, ax_act]:
        ax.set_xlabel('Unseen Test Sample Index', fontsize=16)
        for spine in ax.spines.values(): spine.set_linewidth(2.5); spine.set_color('black')
        ax.tick_params(axis='both', which='major', labelsize=14, width=2.5, length=8)
        ax.tick_params(axis='both', which='minor', width=1.5, length=4)
        for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_fontweight('black')
        ax.grid(True, which="major", ls="-", alpha=0.3, color='black', lw=1.5)
        ax.grid(True, which="minor", ls="--", alpha=0.1, color='black', lw=1.0)
        leg = ax.legend(fontsize=14, loc='best')
        leg.get_frame().set_edgecolor('black'); leg.get_frame().set_linewidth(2.0)
        for t in leg.get_texts(): t.set_fontweight('black')

    plt.suptitle(f'Compressor Global Optimization: Time, Efficiency, and {act_name} Error', fontsize=28, fontweight='black', y=1.05)
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close(fig)


def run_benchmark_only():
    print("\n=============================================")
    print("   启动压缩机系统 AI vs 传统 Benchmark (断点续传)")
    print("=============================================\n")

    try:
        BASE_DIR = Path(__file__).resolve().parent
    except NameError:
        BASE_DIR = Path.cwd()

    model_dir = BASE_DIR.parent / "saved_models"
    dataset_dir = BASE_DIR / "未参加训练数据集"

    output_csv_path = BASE_DIR / 'Compressor_Benchmark_FullData.csv'
    output_fig_path = BASE_DIR / 'Compressor_Benchmark_3Panels.png'

    NUM_SAMPLES = 50
    all_csv_absolute_paths = [str(p.resolve()) for p in dataset_dir.glob("*.csv") if not p.name.startswith("._")]
    test_df = load_and_sample_unseen_data(all_csv_absolute_paths, sample_count=NUM_SAMPLES)

    if test_df.empty:
        print("[❌ 错误] 测试数据集为空，请检查数据文件夹和文件名！")
        return

    t_pow, t_bp = test_df['Power_total_W'].values, test_df['Pout_s4_bar'].values
    flags = test_df[['Flag_S1_Active', 'Flag_S2_Active', 'Flag_S3_Active', 'Flag_S4_Active']].values
    t_spd = test_df[['n_s1', 'n_s2', 'n_s3', 'n_s4']].values
    t_eff = test_df['eta_ex_system'].values
    t_gc = test_df['Gc_kg_s'].values

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[加载模型] 硬件分配至: {device}")

    sX = joblib.load(str(model_dir / 'scaler_X.pkl'))
    sY = joblib.load(str(model_dir / 'scaler_Y.pkl'))

    entire_model_save_path = str(model_dir / 'compressor_inverse_mlp.pkl')
    loaded_inverse_wrapper = joblib.load(entire_model_save_path)
    base_pytorch_model = loaded_inverse_wrapper.model
    mlp = base_pytorch_model.to(device)
    mlp.eval()
    ai_model = TorchMLPWrapper(mlp, device)

    # --- 初始化内存状态记录表 ---
    ai_time_list, trad_time_list = [], []
    ai_eff_error_list, trad_eff_error_list = [], []
    ai_spd_error_list, trad_spd_error_list = [], []
    benchmark_records = []
    start_index = 0

    # ====================================================
    # 核心增强：断点续传（Auto-Resume）内存恢复逻辑
    # ====================================================
    if output_csv_path.exists():
        try:
            existing_df = pd.read_csv(output_csv_path)
            start_index = len(existing_df)
            if 0 < start_index < NUM_SAMPLES:
                print(f"[♻️ 自动恢复] 检测到上次已计算 {start_index} 个点，正在恢复内存状态...")
                benchmark_records = existing_df.to_dict('records')
                # 精准对齐 CSV 列名
                ai_time_list = existing_df['AI_Exec_Time_s'].tolist()
                trad_time_list = existing_df['Trad_Exec_Time_s'].tolist()
                ai_eff_error_list = existing_df['AI_Eff_Absolute_Error'].tolist()
                trad_eff_error_list = existing_df['Trad_Eff_Absolute_Error'].tolist()
                ai_spd_error_list = existing_df['AI_Speed_MAE'].tolist()
                trad_spd_error_list = existing_df['Trad_Speed_MAE'].tolist()
            elif start_index >= NUM_SAMPLES:
                print(f"[✅ 任务已完成] 检测到 {start_index} 个点，达到或超过目标数量 {NUM_SAMPLES}，无需重复计算。")
                return
        except Exception as e:
            print(f"[⚠️ 警告] 读取历史数据失败 ({e})，将从头开始计算。")
            start_index = 0

    print(f"\n[测试启动] 展开 {NUM_SAMPLES} 个样本的瞬态对决，当前从第 {start_index + 1} 个点开始...\n")

    for i in range(start_index, NUM_SAMPLES):
        current_flags = flags[i]
        active_mask = current_flags == 1.0

        # --- 阶段 A：AI 模型瞬态推理 ---
        current_input_features = np.array([
            t_pow[i], t_bp[i], current_flags[0], current_flags[1], current_flags[2], current_flags[3]
        ]).reshape(1, -1)

        t0_ai = time.perf_counter()
        raw_scaled_prediction = ai_model.predict(sX.transform(current_input_features))
        y_pred_physical = sY.inverse_transform(raw_scaled_prediction)[0]
        ai_execution_time = time.perf_counter() - t0_ai

        ai_pred_speeds = y_pred_physical[0:4]
        ai_pred_efficiency = y_pred_physical[4]

        ai_eff_error = abs(t_eff[i] - ai_pred_efficiency)
        ai_spd_error = mean_absolute_error(t_spd[i][active_mask], ai_pred_speeds[active_mask]) if np.sum(
            active_mask) > 0 else 0.0

        ai_time_list.append(ai_execution_time)
        ai_eff_error_list.append(ai_eff_error)
        ai_spd_error_list.append(ai_spd_error)

        # --- 阶段 B：传统 SLSQP 迭代寻优 ---
        print(f"   ► 正在计算第 {i + 1}/{NUM_SAMPLES} 个点，LHS 矩阵扫描中...")
        t0_trad = time.perf_counter()

        opt_spd, opt_gc, opt_eff, is_success = global_optimization_traditional(
            target_power=t_pow[i], target_pressure=t_bp[i], active_flags=current_flags,
            base_des_para=configure_design_parameters(), n_restarts=1, seed=42
        )
        trad_execution_time = time.perf_counter() - t0_trad
        trad_time_list.append(trad_execution_time)

        if opt_eff <= 0.01 or not is_success:
            print(f"      [✖] 优化器陷入物理死区 (耗时 {trad_execution_time:.2f}s)。")
            trad_eff_error_list.append(np.nan)
            trad_spd_error_list.append(np.nan)
        else:
            trad_eff_error = abs(t_eff[i] - opt_eff)
            trad_spd_error = mean_absolute_error(opt_spd[active_mask], t_spd[i][active_mask]) if np.sum(
                active_mask) > 0 else 0.0
            trad_eff_error_list.append(trad_eff_error)
            trad_spd_error_list.append(trad_spd_error)

        # --- 阶段 C：构建字典并流式存储 ---
        record_dict = {
            'Sample_ID': i + 1, 'Target_Power_W': t_pow[i], 'Target_BP_bar': t_bp[i],
            'Stage_Active_Mode': "-".join([str(idx + 1) for idx, val in enumerate(current_flags) if val == 1.0]),
            'True_Speed_s1': t_spd[i][0], 'True_Speed_s2': t_spd[i][1], 'True_Speed_s3': t_spd[i][2],
            'True_Speed_s4': t_spd[i][3],
            'True_Gc_kg_s': t_gc[i], 'True_Efficiency': t_eff[i],
            'AI_Speed_s1': ai_pred_speeds[0], 'AI_Speed_s2': ai_pred_speeds[1], 'AI_Speed_s3': ai_pred_speeds[2],
            'AI_Speed_s4': ai_pred_speeds[3],
            'AI_Efficiency': ai_pred_efficiency, 'AI_Exec_Time_s': ai_execution_time,
            'AI_Speed_MAE': ai_spd_error, 'AI_Eff_Absolute_Error': ai_eff_error,
            'Trad_Speed_s1': opt_spd[0] if is_success else np.nan,
            'Trad_Speed_s2': opt_spd[1] if is_success else np.nan,
            'Trad_Speed_s3': opt_spd[2] if is_success else np.nan,
            'Trad_Speed_s4': opt_spd[3] if is_success else np.nan,
            'Trad_Gc_kg_s': opt_gc if is_success else np.nan, 'Trad_Efficiency': opt_eff if is_success else np.nan,
            'Trad_Exec_Time_s': trad_execution_time,
            'Trad_Speed_MAE': trad_spd_error if is_success else np.nan,
            'Trad_Eff_Absolute_Error': trad_eff_error if is_success else np.nan, 'Trad_Converged': is_success
        }
        benchmark_records.append(record_dict)

        # 落盘保存 CSV
        pd.DataFrame(benchmark_records).to_csv(output_csv_path, index=False, encoding='utf-8-sig')

        # 实时生成性能对比图表
        visualize_benchmark_3panels(
            ai_time_list, trad_time_list, ai_eff_error_list, trad_eff_error_list,
            ai_spd_error_list, trad_spd_error_list, "Speed Ratio", "Non-dimensional", output_fig_path
        )

        print(f"   [💾 实时存档] 第 {i + 1}/{NUM_SAMPLES} 个点计算完毕，CSV与图片已覆盖更新！")

    print("\n[🏆 完美收官] 所有测试点计算完毕，对抗测试结束！")


if __name__ == "__main__":
    run_benchmark_only()
