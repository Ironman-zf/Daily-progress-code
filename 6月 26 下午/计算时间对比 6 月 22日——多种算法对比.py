import os
import re
import time
import copy
import joblib
import warnings
import pandas as pd
import numpy as np
from types import SimpleNamespace
from scipy.optimize import minimize, differential_evolution, Bounds, NonlinearConstraint
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from pathlib import Path
import torch
import torch.nn as nn
from scipy.stats import qmc
from compression_system1 import compression_system4

class TorchMLPWrapper:
    """
    封装类：对外提供与 Sklearn 一致的 API，不干扰底部的画图和导出代码
    """

    def __init__(self, pytorch_model, compute_device):
        self.model = pytorch_model
        self.device = compute_device

    def predict(self, input_features):
        self.model.eval()
        with torch.no_grad():
            feature_tensor = torch.tensor(input_features, dtype=torch.float32).to(self.device)
            prediction_tensor = self.model(feature_tensor)
            prediction_numpy = prediction_tensor.cpu().numpy()
        return prediction_numpy


class DynamicMLP(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_layers):
        super(DynamicMLP, self).__init__()
        network_layers = []
        current_dimension = input_dimension

        for hidden_dimension in hidden_layers:
            network_layers.append(nn.Linear(current_dimension, hidden_dimension))
            network_layers.append(nn.ReLU())
            current_dimension = hidden_dimension

        network_layers.append(nn.Linear(current_dimension, output_dimension))
        self.network = nn.Sequential(*network_layers)

    def forward(self, x):
        return self.network(x)
class PassThroughScaler:
    def fit_transform(self, Y):
        return np.array(Y, copy=True)

    def transform(self, Y):
        return np.array(Y, copy=True)

    def inverse_transform(self, Y_scaled):
        return np.array(Y_scaled, copy=True)
# ==========================================
# 1. 您的严格物理模型定义区
# ==========================================
class InfeasibleDesignError(Exception):
    def __init__(self, message, stage):
        self.stage = stage
        self.message = message
        self.full_message = f"[致命物理冲突 - 发生于第 {stage} 级] {message}"
        super().__init__(self.full_message)


# ==========================================
# 2. AI 基础设施与工具函数
# ==========================================
class PhysicsAwareScaler:
    def __init__(self): self.scaler = StandardScaler()

    def fit_transform(self, X): X_out = np.array(X, copy=True); X_out[:, 0:2] = self.scaler.fit_transform(
        X_out[:, 0:2]); return X_out

    def transform(self, X): X_out = np.array(X, copy=True); X_out[:, 0:2] = self.scaler.transform(
        X_out[:, 0:2]); return X_out

    def inverse_transform(self, X_scaled): X_out = np.array(X_scaled, copy=True); X_out[
        :, 0:2] = self.scaler.inverse_transform(X_out[:, 0:2]); return X_out


class PhysicsEmbeddedMLP(nn.Module):
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
        masked_speeds = raw_output[:, 0:4] * flags + 1.0 * (1.0 - flags)
        efficiency = raw_output[:, 4].unsqueeze(1)
        return torch.cat((masked_speeds, efficiency), dim=1)


class TorchMLPWrapper:
    def __init__(self, pytorch_model, compute_device): self.model = pytorch_model; self.device = compute_device

    def predict(self, input_features):
        self.model.eval()
        with torch.no_grad(): return self.model(
            torch.tensor(input_features, dtype=torch.float32).to(self.device)).cpu().numpy()


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


def load_and_sample_unseen_data(unseen_file_paths, sample_count=50):
    valid_dataframes_list = []
    for fp in unseen_file_paths:
        if not os.path.exists(fp): continue
        raw_df = pd.read_csv(fp)
        if 'Exit_Flag' in raw_df.columns: raw_df = raw_df[raw_df['Exit_Flag'] == 1]
        clean_df = raw_df.dropna(how='all').dropna(axis=1, how='all').dropna(how='any').copy()
        if clean_df.empty: continue

        possible_power_columns = ['Power_total_W', 'Power_in_total_W', 'Power_out_total_W', 'W_total']
        actual_power_col = next((col for col in possible_power_columns if col in clean_df.columns), None)
        ref_col = 'Gc_kg_s' if 'Gc_kg_s' in clean_df.columns else actual_power_col
        if ref_col is None: continue

        df_unique = clean_df.drop_duplicates(subset=[ref_col]).sort_values(by=ref_col).reset_index(drop=True)

        if 'Gc_kg_s' in df_unique.columns and actual_power_col is not None:
            max_p_idx = df_unique[actual_power_col].idxmax()
            df_valid = df_unique.iloc[:max_p_idx + 1].copy()
        else:
            df_valid = df_unique.copy()

        bp = float(m.group(1)) if (m := re.search(r'-(\d+)-compressor', os.path.basename(fp))) else 0.0
        acts = [int(s) for s in os.path.basename(fp).split('-' + str(int(bp)))[0].split('-') if s.isdigit()]

        df_valid['Inlet_Pressure_bar'] = bp
        df_valid['Flag_S1_Active'] = 1.0 if 1 in acts else 0.0
        df_valid['Flag_S2_Active'] = 1.0 if 2 in acts else 0.0
        df_valid['Flag_S3_Active'] = 1.0 if 3 in acts else 0.0
        df_valid['Flag_S4_Active'] = 1.0 if 4 in acts else 0.0
        valid_dataframes_list.append(df_valid)

    if not valid_dataframes_list: return pd.DataFrame()
    master_df = pd.concat(valid_dataframes_list, axis=0, ignore_index=True)
    return master_df.sample(n=min(sample_count, len(master_df)), random_state=42).reset_index(drop=True)


# ==========================================
# 3. 多态全局优化算法 (SLSQP / TRUST / DE)
# ==========================================
def thermodynamic_forward_model(n_array, mass_flow, base_des_para):
    ac_local = SimpleNamespace(Gc=mass_flow, n_=n_array, T_tank_out=298.15)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = compression_system4(base_des_para, ac_local)

            # 【物理机理修正】基于梯度的优化器严禁返回 np.nan。必须返回稳定的物理极值作为边界惩罚
            if not out.get('feasible', False):
                return 1e7, 1e7, 0.0

            power = out.get('Power_in_total_W', 1e7)
            p_comp_array = out.get('P_air_comp', [])
            pressure = p_comp_array[-1] if len(p_comp_array) > 0 else 1e7
            eff = out.get('eta_ex_system', 0.0)

            # 再次检查数值合法性，防止偶发性上层崩溃
            if not (np.isfinite(power) and np.isfinite(pressure) and np.isfinite(eff)):
                return 1e7, 1e7, 0.0

            return power, pressure, eff
    except Exception:
        return 1e7, 1e7, 0.0


def global_optimization_polymorphic(target_power, target_pressure, active_flags, base_des_para,
                                    optimizer_type='SLSQP', n_restarts=20, seed=42, power_tol=0.001, bp_tol=0.001):
    rng = np.random.default_rng(seed)
    last_x, last_res = None, (np.nan, np.nan, np.nan)

    def eval_phys(x):
        nonlocal last_x, last_res
        if last_x is None or not np.array_equal(x, last_x):
            last_res = thermodynamic_forward_model(x[0:4], x[4], base_des_para)
            last_x = np.copy(x)
        return last_res

    def obj(x):
        return -eval_phys(x)[2]

    def safe_rel_err(val, target):
        if not np.isfinite(val) or not np.isfinite(target) or abs(target) < 1e-12: return -1e9
        return (val - target) / target

    def c_pow(x):
        return safe_rel_err(eval_phys(x)[0], target_power)

    def c_pres(x):
        return safe_rel_err(eval_phys(x)[1], target_pressure)

    bounds_spd = [(0.7, 1.1) if f == 1.0 else (1.0, 1.0) for f in active_flags]
    gc_bounds = (3.0, 25.0)
    full_bounds = bounds_spd + [gc_bounds]

    best_res, best_eff = None, -np.inf

    # ================= 分支 1：DE 无导数全局寻优 =================
    if optimizer_type == 'DE':
        nlc_pow = NonlinearConstraint(c_pow, -power_tol, power_tol)
        nlc_pres = NonlinearConstraint(c_pres, -bp_tol, bp_tol)
        try:
            res = differential_evolution(
                func=obj, bounds=full_bounds, constraints=(nlc_pow, nlc_pres),
                strategy='best1bin', maxiter=20, popsize=10, seed=seed, disp=False
            )
            if res.success and np.isfinite(res.fun) and (-res.fun > 0.01):
                return res.x[0:4], res.x[4], -res.fun, True
        except Exception:
            pass
        return np.full(4, np.nan), np.nan, np.nan, False

    # ================= 分支 2：基于梯度的多起点算法 =================
    lower_bounds = np.array([b[0] for b in full_bounds])
    upper_bounds = np.array([b[1] for b in full_bounds])

    n_lhs_samples = n_restarts - 1
    lhs_physical_samples = None
    if n_lhs_samples > 0:
        lhs_sampler = qmc.LatinHypercube(d=len(full_bounds), seed=seed)
        lhs_physical_samples = lower_bounds + lhs_sampler.random(n=n_lhs_samples) * (upper_bounds - lower_bounds)

    for k in range(n_restarts):
        current_x0 = np.array([1.0, 1.0, 1.0, 1.0, 10.138]) if k == 0 else np.copy(lhs_physical_samples[k - 1])

        try:
            if optimizer_type == 'SLSQP':
                constraints = [
                    {'type': 'ineq', 'fun': lambda x: power_tol - c_pow(x)},
                    {'type': 'ineq', 'fun': lambda x: c_pow(x) + power_tol},
                    {'type': 'ineq', 'fun': lambda x: bp_tol - c_pres(x)},
                    {'type': 'ineq', 'fun': lambda x: c_pres(x) + bp_tol}
                ]
                res = minimize(obj, x0=current_x0, method='SLSQP', bounds=full_bounds,
                               constraints=constraints,
                               options={'ftol': 1e-5, 'maxiter': 666, 'eps': 1e-3, 'disp': False})

            elif optimizer_type == 'TRUST':
                bnds = Bounds(lower_bounds, upper_bounds)
                nlc_pow = NonlinearConstraint(c_pow, -power_tol, power_tol)
                nlc_pres = NonlinearConstraint(c_pres, -bp_tol, bp_tol)
                res = minimize(obj, x0=current_x0, method='trust-constr', bounds=bnds,
                               constraints=(nlc_pow, nlc_pres), options={'xtol': 1e-4, 'maxiter': 666, 'disp': False})

            if res.success and np.isfinite(res.fun) and (-res.fun > 0.01):
                if -res.fun > best_eff:
                    best_eff = -res.fun
                    best_res = res
        except Exception:
            continue

    if best_res is None:
        return np.full(4, np.nan), np.nan, np.nan, False

    return best_res.x[0:4], best_res.x[4], -best_res.fun, True


# ==========================================
# 4. 可视化出图
# ==========================================
def visualize_benchmark_3panels(ai_t, trad_t, ai_eff_err, trad_eff_err, ai_act_err, trad_act_err, act_name, act_unit,
                                algo_name, filename):
    plt.rcParams['font.weight'] = 'black';
    plt.rcParams['axes.labelweight'] = 'black';
    plt.rcParams['axes.titleweight'] = 'black'
    fig, (ax_time, ax_eff, ax_act) = plt.subplots(1, 3, figsize=(28, 8))
    idx = np.arange(1, len(ai_t) + 1)

    ax_time.plot(idx, trad_t, 's-', color='#d62728', lw=2, ms=8, alpha=0.8, label=f'Traditional ({algo_name})')
    ax_time.plot(idx, ai_t, 'o-', color='#1f77b4', lw=2, ms=8, alpha=0.8, label='AI (PyTorch MLP)')
    ax_time.set_title('Computational Time per Sample', fontsize=20, pad=15)
    ax_time.set_ylabel('Execution Time (Seconds)', fontsize=16)

    safe_trad_eff = np.where(np.isnan(trad_eff_err), np.nan, np.clip(trad_eff_err, 1e-6, None))
    safe_ai_eff = np.clip(ai_eff_err, 1e-6, None)
    ax_eff.plot(idx, safe_trad_eff, 's-', color='#d62728', lw=2, ms=8, alpha=0.8, label=f'Traditional ({algo_name})')
    ax_eff.plot(idx, safe_ai_eff, 'o-', color='#1f77b4', lw=2, ms=8, alpha=0.8, label='AI (PyTorch MLP)')
    ax_eff.set_title('Efficiency Error', fontsize=20, pad=15)
    ax_eff.set_ylabel('Absolute Error', fontsize=16)

    ax_act.plot(idx, trad_act_err, 's-', color='#d62728', lw=2, ms=8, alpha=0.8, label=f'Traditional ({algo_name})')
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
        leg.get_frame().set_edgecolor('black');
        leg.get_frame().set_linewidth(2.0)
        for t in leg.get_texts(): t.set_fontweight('black')

    plt.suptitle(f'Compressor Optimization: AI vs {algo_name}', fontsize=28, fontweight='black', y=1.05)
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close(fig)


# ==========================================
# 5. 主程序控制流 (循环多算法评测)
# ==========================================
def run_benchmark_multi_algo():
    print("\n=============================================")
    print("   启动多态寻优对比测试 (AI vs SLSQP/TRUST/DE)")
    print("=============================================\n")

    try:
        BASE_DIR = Path(__file__).resolve().parent
    except NameError:
        BASE_DIR = Path.cwd()

    model_dir = BASE_DIR.parent / "saved_models1"
    dataset_dir = BASE_DIR / "未参加训练数据集_clean"

    NUM_SAMPLES = 50
    all_csv_absolute_paths = [str(p.resolve()) for p in dataset_dir.glob("*.csv") if not p.name.startswith("._")]
    test_df = load_and_sample_unseen_data(all_csv_absolute_paths, sample_count=NUM_SAMPLES)

    if test_df.empty:
        print("[❌ 错误] 测试数据集为空，请检查数据文件夹和文件名！")
        return

    possible_power_columns = ['Power_total_W', 'Power_in_total_W', 'Power_out_total_W', 'W_total']
    actual_power_col = next((col for col in possible_power_columns if col in test_df.columns), None)

    t_pow = test_df[actual_power_col].values
    t_bp = test_df['Pout_s4_bar'].values
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
    mlp = loaded_inverse_wrapper.model.to(device)
    mlp.eval()
    ai_model = TorchMLPWrapper(mlp, device)

    # 🚨 核心逻辑：遍历三种不同的传统优化算法
    TARGET_ALGORITHMS = [ 'TRUST']

    for current_algo in TARGET_ALGORITHMS:
        print(f"\n=============================================")
        print(f"   ▶ 开始测评算法轨道: {current_algo}")
        print(f"=============================================")

        # 文件名相互独立隔离
        output_csv_path = BASE_DIR / f'Compressor_Benchmark_{current_algo}.csv'
        output_fig_path = BASE_DIR / f'Compressor_Benchmark_{current_algo}.png'

        ai_time_list, trad_time_list = [], []
        ai_eff_error_list, trad_eff_error_list = [], []
        ai_spd_error_list, trad_spd_error_list = [], []
        benchmark_records = []
        start_index = 0

        if output_csv_path.exists():
            try:
                existing_df = pd.read_csv(output_csv_path)
                start_index = len(existing_df)
                if 0 < start_index < NUM_SAMPLES:
                    print(f"[♻️ 自动恢复] 检测到上次已计算 {start_index} 个点，正在恢复内存状态...")
                    benchmark_records = existing_df.to_dict('records')
                    ai_time_list = existing_df['AI_Exec_Time_s'].tolist()
                    trad_time_list = existing_df['Trad_Exec_Time_s'].tolist()
                    ai_eff_error_list = existing_df['AI_Eff_Absolute_Error'].tolist()
                    trad_eff_error_list = existing_df['Trad_Eff_Absolute_Error'].tolist()
                    ai_spd_error_list = existing_df['AI_Speed_MAE'].tolist()
                    trad_spd_error_list = existing_df['Trad_Speed_MAE'].tolist()
                elif start_index >= NUM_SAMPLES:
                    print(f"[✅ 任务已完成] 算法 {current_algo} 均已测试完毕，自动跳过。")
                    continue
            except Exception as e:
                start_index = 0

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

            # --- 阶段 B：多态统迭代寻优 ---
            restarts_config = {'SLSQP': 50, 'TRUST': 5, 'DE': 50}
            current_restarts = restarts_config[current_algo]

            print(f"   ► [{current_algo}] 正在计算第 {i + 1}/{NUM_SAMPLES} 个点 (重启配置: {current_restarts}) ...")
            t0_trad = time.perf_counter()

            opt_spd, opt_gc, opt_eff, is_success = global_optimization_polymorphic(
                target_power=t_pow[i], target_pressure=t_bp[i], active_flags=current_flags,
                base_des_para=configure_design_parameters(), optimizer_type=current_algo, n_restarts=current_restarts,
                seed=42
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
                'Sample_ID': i + 1, 'Algorithm': current_algo,
                'Target_Power_W': t_pow[i], 'Target_BP_bar': t_bp[i],
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

            pd.DataFrame(benchmark_records).to_csv(output_csv_path, index=False, encoding='utf-8-sig')

            visualize_benchmark_3panels(
                ai_time_list, trad_time_list, ai_eff_error_list, trad_eff_error_list,
                ai_spd_error_list, trad_spd_error_list, "Speed Ratio", "Non-dimensional", current_algo, output_fig_path
            )


if __name__ == "__main__":
    run_benchmark_multi_algo()