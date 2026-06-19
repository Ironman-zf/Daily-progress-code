import os
import re
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.interpolate import PchipInterpolator

# 锁定全局随机种子
np.random.seed(42)
torch.manual_seed(42)



# 归一化数据
class PhysicsAwareScaler:
    """
    X 缩放器：对连续物理量（功率、背压）归一化，不对 0/1 开关归一化
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
    Y 缩放器（直通车）：因为转速比 [0.7, 1.1] 和效率 [0.8, 0.9] 已经是完美的网络目标，
    禁止任何数学平移，保护非激活级的物理绝对值 1.0。
    """

    def fit_transform(self, Y): return np.array(Y, copy=True)

    def transform(self, Y): return np.array(Y, copy=True)

    def inverse_transform(self, Y_scaled): return np.array(Y_scaled, copy=True)


def parse_physics_from_filename(filename):
    pure_name = os.path.basename(filename)
    back_pressure_match = re.search(r'-(80|90|100)-compressor', pure_name)
    back_pressure_bar = float(back_pressure_match.group(1)) if back_pressure_match else 0.0

    mode_string = pure_name.split('-' + str(int(back_pressure_bar)))[0]
    active_stages = [int(s) for s in mode_string.split('-') if s.isdigit()]

    return (back_pressure_bar,
            1.0 if 1 in active_stages else 0.0,
            1.0 if 2 in active_stages else 0.0,
            1.0 if 3 in active_stages else 0.0,
            1.0 if 4 in active_stages else 0.0)
def load_and_unify_all_datasets(file_list, uniform_target_points=288):
    unified_dataframe_list = []
    for file in file_list:
        if not os.path.exists(file): continue
        raw_df = pd.read_csv(file)
        dense_df = raw_df.dropna(how='all').dropna(axis=1, how='all').dropna(how='any').copy()
        bp, flag_1, flag_2, flag_3, flag_4 = parse_physics_from_filename(file)

        dense_df['Back_Pressure_bar'] = bp
        dense_df['Flag_S1_Active'] = flag_1;
        dense_df['Flag_S2_Active'] = flag_2
        dense_df['Flag_S3_Active'] = flag_3;
        dense_df['Flag_S4_Active'] = flag_4
        unified_dataframe_list.append(dense_df)

    return pd.concat(unified_dataframe_list, axis=0, ignore_index=True)


def create_universal_features_and_targets(master_dataframe, design_constants):
    rel_power = master_dataframe['Power_total_W'].values / design_constants['power_w']
    back_pressures = master_dataframe['Back_Pressure_bar'].values / design_constants['Back_Pressure_bar']
    flag_s1 = master_dataframe['Flag_S1_Active'].values
    flag_s2 = master_dataframe['Flag_S2_Active'].values
    flag_s3 = master_dataframe['Flag_S3_Active'].values
    flag_s4 = master_dataframe['Flag_S4_Active'].values

    universal_X = np.column_stack((rel_power, back_pressures, flag_s1, flag_s2, flag_s3, flag_s4))

    speed_s1 = master_dataframe['n_s1'].values
    speed_s2 = master_dataframe['n_s2'].values
    speed_s3 = master_dataframe['n_s3'].values
    speed_s4 = master_dataframe['n_s4'].values
    system_efficiency = master_dataframe['eta_ex_system'].values

    universal_Y = np.column_stack((speed_s1, speed_s2, speed_s3, speed_s4, system_efficiency))
    return universal_X, universal_Y

def keep_high_eta_per_power(df,power_col='Power_total_W',eff_col='eta_ex_system',group_cols=None,
                            power_tol=1e-3):

    if group_cols is None:
        group_cols = ['Back_Pressure_bar','Flag_S1_Active','Flag_S2_Active','Flag_S3_Active','Flag_S4_Active']
    kept_groups = []
    for _, g in df.groupby(group_cols):
        g = g.copy()
        # 将功率按容差分箱，避免浮点数导致“本应相同却分不开”
        g['_power_bin_'] = np.round(g[power_col] / power_tol) * power_tol
        # 每个功率桶内，只保留效率最高的那个点
        idx = g.groupby('_power_bin_')[eff_col].idxmax()
        g_kept = g.loc[idx].copy()
        # 按功率排序，方便后续画图或训练
        g_kept = g_kept.sort_values(power_col).reset_index(drop=True)
        # 删除临时列
        g_kept.drop(columns=['_power_bin_'], inplace=True)
        kept_groups.append(g_kept)

    return pd.concat(kept_groups, axis=0, ignore_index=True)

class TorchMLPWrapper:
    def __init__(self, pytorch_model, compute_device):
        self.model = pytorch_model
        self.device = compute_device

    def predict(self, input_features):
        self.model.eval()
        with torch.no_grad():
            feature_tensor = torch.tensor(input_features, dtype=torch.float32).to(self.device)
            return self.model(feature_tensor).cpu().numpy()


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

        # 核心物理截断：(预测值 * 激活) + (1.0 * 不激活)
        masked_speeds = raw_output[:, 0:4] * flags + 1.0 * (1.0 - flags)

        efficiency = raw_output[:, 4].unsqueeze(1)
        return torch.cat((masked_speeds, efficiency), dim=1)


def physical_masked_loss(predictions, targets, flags):
    """
    只惩罚被激活的压缩级，避免网络分心去学 1.0
    """
    pred_spd = predictions[:, 0:4];
    true_spd = targets[:, 0:4]
    pred_eff = predictions[:, 4];
    true_eff = targets[:, 4]

    spd_squared_error = (pred_spd - true_spd) ** 2
    active_spd_error = spd_squared_error * flags
    active_count = flags.sum() + 1e-8

    spd_loss = active_spd_error.sum() / active_count
    eff_loss = nn.functional.mse_loss(pred_eff, true_eff)
    return spd_loss + 8*eff_loss


def execute_pytorch_training(X_data, Y_data, hidden_layers, learning_rate, total_epochs, batch_size, compute_device):
    total_samples = len(X_data)
    validation_size = int(total_samples * 0.1)
    train_size = total_samples - validation_size

    features_tensor = torch.tensor(X_data, dtype=torch.float32).to(compute_device)
    targets_tensor = torch.tensor(Y_data, dtype=torch.float32).to(compute_device)

    full_dataset = TensorDataset(features_tensor, targets_tensor)
    train_dataset, validation_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(42)
    )

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    neural_network_model = PhysicsEmbeddedMLP(X_data.shape[1], Y_data.shape[1], hidden_layers).to(compute_device)
    optimizer = optim.AdamW(neural_network_model.parameters(), lr=learning_rate, weight_decay=0.001)
    learning_rate_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_validation_loss = float('inf')
    best_model_weights = None
    epochs_without_improvement = 0
    patience_limit = 200

    for epoch in range(total_epochs):
        neural_network_model.train()
        for batch_features, batch_targets in train_data_loader:
            optimizer.zero_grad()
            predictions = neural_network_model(batch_features)
            batch_flags = batch_features[:, 2:6]

            loss = physical_masked_loss(predictions, batch_targets, batch_flags)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(neural_network_model.parameters(), max_norm=1.0)
            optimizer.step()

        neural_network_model.eval()
        current_validation_loss = 0.0
        with torch.no_grad():
            for val_features, val_targets in validation_data_loader:
                val_predictions = neural_network_model(val_features)
                val_flags = val_features[:, 2:6]
                val_batch_loss = physical_masked_loss(val_predictions, val_targets, val_flags)
                current_validation_loss += val_batch_loss.item() * val_features.size(0)

        current_validation_loss /= max(validation_size, 1)
        learning_rate_scheduler.step(current_validation_loss)

        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            best_model_weights = copy.deepcopy(neural_network_model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience_limit:
            print(f"      [早停触发] 第 {epoch + 1} 轮停止迭代。最佳验证损失: {best_validation_loss:.6f}")
            break

    if best_model_weights is not None:
        neural_network_model.load_state_dict(best_model_weights)

    return neural_network_model

def train_mlp_with_gaussian_hard_mining(X_train_scaled,Y_train_scaled,scaler_X,scaler_Y,design_constants,
        duplication_factor=5):

    active_device = torch.device("mps"if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available()
        else "cpu")

    print(f"\n[硬件状态] 深度学习计算后端已锁定为: {active_device}")
    # =====================================================
    # 第一阶段
    # =====================================================
    print(f"--- 启动第一阶段：基于 {len(X_train_scaled)} 样本集的 PyTorch 基础探索 ---")
    base_model = execute_pytorch_training(X_train_scaled,Y_train_scaled,
        (512,256,128,64),0.002,1000,1024,active_device)
    # =====================================================
    # 第二阶段
    # =====================================================
    print("--- 启动第二阶段：困难样本评估 ---")
    base_model.eval()
    with torch.no_grad():
        X_eval_tensor = torch.tensor(X_train_scaled,dtype=torch.float32).to(active_device)
        Y_pred_scaled = base_model(X_eval_tensor).cpu().numpy()
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    Y_true = scaler_Y.inverse_transform(Y_train_scaled)
    # =====================================================
    # 用系统㶲效率误差作为困难度
    # =====================================================
    eff_error = np.abs(Y_true[:,4] -Y_pred[:,4])

    hard_threshold = np.percentile(eff_error,90 )
    hard_mask = eff_error >= hard_threshold

    X_hard = X_train_scaled[hard_mask]
    Y_hard = Y_train_scaled[hard_mask]
    print(f"[!] 锁定困难样本 {len(X_hard)} 个 ",f"(Top 10% Efficiency Error)")

    # =====================================================
    # 导出困难样本
    # =====================================================

    EXPORT_THRESHOLD = 0.01
    export_mask = eff_error > EXPORT_THRESHOLD
    if np.sum(export_mask) > 0:
        X_real = scaler_X.inverse_transform(X_train_scaled)
        export_X = X_real[export_mask]
        export_Y_true = Y_true[export_mask]
        export_Y_pred = Y_pred[export_mask]
        export_eff_error = eff_error[export_mask]
        outlier_rows = []
        for i in range(len(export_X)):
            flags = export_X[i,2:6]
            active_stages = [str(j+1)
                for j,f in enumerate(flags)
                if float(f) > 0.5]
            outlier_rows.append({"Regulation_Mode":"-".join(active_stages),
            "Back_Pressure_Bar":float(export_X[i,1]),"Relative_Power":float(export_X[i,0]),
            "True_Efficiency":float(export_Y_true[i,4]),
            "Pred_Efficiency":float(export_Y_pred[i,4]),
            "Efficiency_Error":float(export_eff_error[i])})
        df_out = pd.DataFrame(outlier_rows)
        df_out = df_out.sort_values(by="Efficiency_Error",ascending=False)
        export_path = ("Compressor_High_Error_Efficiency_Samples.csv")
        df_out.to_csv(export_path,index=False)
        print(f"[💾] 导出困难工况 "f"{len(df_out)} 条 -> {export_path}")
    # ====================================================
    # 第三阶段
    # =====================================================
    print(f"--- 启动第三阶段："f"{duplication_factor}倍高斯流形扩增 ---")
    X_aug_list = [X_train_scaled]
    Y_aug_list = [Y_train_scaled]
    for _ in range(duplication_factor):
        noise = np.random.normal(0,0.01,size=X_hard.shape)
        # 不扰动开关位
        noise[:,2:6] = 0
        X_aug_list.append(X_hard + noise)
        Y_aug_list.append(Y_hard)
    X_train_aug = np.vstack(X_aug_list)
    Y_train_aug = np.vstack(Y_aug_list)
    print(f"扩增后样本数 = "f"{len(X_train_aug)}")
    # =====================================================
    # 第四阶段
    # =====================================================
    print("--- 启动第四阶段：终极网络训练 ---")
    final_model = execute_pytorch_training(X_train_aug,Y_train_aug,(512,256,128,64),
        0.0005,5000,1024,active_device)
    print("[✔] PyTorch终极反问题网络训练完成！")
    return TorchMLPWrapper(final_model,active_device)
def calculate_and_visualize_model_accuracy_strict_physics(MLP_model, X_test_unscaled, X_test_scaled_data,
                                                          Y_test_real_data, scaler_for_Y):
    Y_predicted_scaled = MLP_model.predict(X_test_scaled_data)
    Y_predicted_real = scaler_for_Y.inverse_transform(Y_predicted_scaled)
    physical_flags = X_test_unscaled[:, 2:6]
    figure, axes_array = plt.subplots(nrows=2, ncols=3, figsize=(20, 13))
    compressor_stages = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
    plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    metric_box_style = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='black')
    for stage_index in range(4):
        row_idx = stage_index // 3;
        col_idx = stage_index % 3
        current_axis = axes_array[row_idx, col_idx]
        active_mask = physical_flags[:, stage_index] == 1.0
        true_active_speeds = Y_test_real_data[active_mask, stage_index]
        predicted_active_speeds = Y_predicted_real[active_mask, stage_index]
        current_axis.scatter(true_active_speeds, predicted_active_speeds, c=plot_colors[stage_index], s=60, alpha=0.7,edgecolors='black', linewidths=1.0)
        if len(true_active_speeds) > 0:
            raw_min = min(np.min(true_active_speeds), np.min(predicted_active_speeds))
            raw_max = max(np.max(true_active_speeds), np.max(predicted_active_speeds))
            margin = max((raw_max - raw_min) * 0.05, 0.01)

            current_axis.plot([raw_min - margin, raw_max + margin], [raw_min - margin, raw_max + margin], color='black',linestyle='--', linewidth=2.5, label='Ideal')

            current_axis.set_xlim([raw_min - margin, raw_max + margin]);
            current_axis.set_ylim([raw_min - margin, raw_max + margin])

            current_r2 = r2_score(true_active_speeds, predicted_active_speeds)
            current_rmse = np.sqrt(mean_squared_error(true_active_speeds, predicted_active_speeds))
            current_mae = mean_absolute_error(true_active_speeds, predicted_active_speeds)

            metric_text = f"R^2 = {current_r2:.4f}\nRMSE = {current_rmse:.4f}\nMAE = {current_mae:.4f}"
            current_axis.text(0.05, 0.95, metric_text, transform=current_axis.transAxes, fontsize=18, fontweight='bold',verticalalignment='top', bbox=metric_box_style)

        current_axis.set_title(f'{compressor_stages[stage_index]} (Active Only)' if len(
            true_active_speeds) > 0 else f'{compressor_stages[stage_index]} (No Active Samples)', fontsize=20,
                               fontweight='bold')
        current_axis.set_xlabel('True Target Speed', fontsize=20, fontweight='bold');
        current_axis.set_ylabel('Predicted Speed', fontsize=20, fontweight='bold')
        current_axis.tick_params(axis='both', which='major', labelsize=12)
        plt.setp(current_axis.get_xticklabels(), fontweight='bold');
        plt.setp(current_axis.get_yticklabels(), fontweight='bold')
        current_axis.grid(True, linestyle='--', alpha=0.7)
        if len(true_active_speeds) > 0: current_axis.legend(prop={'size': 17, 'weight': 'bold'}, loc='lower right')

    eff_axis = axes_array[1, 1]
    true_eff = Y_test_real_data[:, 4];
    predicted_eff = Y_predicted_real[:, 4]
    eff_axis.scatter(true_eff, predicted_eff, c=plot_colors[4], s=60, alpha=0.7, edgecolors='black', linewidths=1.0)

    raw_min_eff = min(np.min(true_eff), np.min(predicted_eff));
    raw_max_eff = max(np.max(true_eff), np.max(predicted_eff))
    eff_margin = max((raw_max_eff - raw_min_eff) * 0.05, 0.005)
    eff_axis.plot([raw_min_eff - eff_margin, raw_max_eff + eff_margin],
                  [raw_min_eff - eff_margin, raw_max_eff + eff_margin], color='black', linestyle='--', linewidth=2.5,
                  label='Ideal')
    eff_axis.set_xlim([raw_min_eff - eff_margin, raw_max_eff + eff_margin]);
    eff_axis.set_ylim([raw_min_eff - eff_margin, raw_max_eff + eff_margin])

    eff_r2 = r2_score(true_eff, predicted_eff)
    eff_rmse = np.sqrt(mean_squared_error(true_eff, predicted_eff))
    eff_mae = mean_absolute_error(true_eff, predicted_eff)
    eff_axis.text(0.05, 0.95, f"R^2 = {eff_r2:.4f}\nRMSE = {eff_rmse:.4f}\nMAE = {eff_mae:.4f}",
                  transform=eff_axis.transAxes, fontsize=18, fontweight='bold', verticalalignment='top',
                  bbox=metric_box_style)
    eff_axis.set_title('System Total Exergy Efficiency', fontsize=19, fontweight='bold');
    eff_axis.set_xlabel('True System Efficiency', fontsize=19, fontweight='bold');
    eff_axis.set_ylabel('AI Predicted Efficiency', fontsize=19, fontweight='bold')
    eff_axis.tick_params(axis='both', which='major', labelsize=12)
    plt.setp(eff_axis.get_xticklabels(), fontweight='bold');
    plt.setp(eff_axis.get_yticklabels(), fontweight='bold')
    eff_axis.grid(True, linestyle='--', alpha=0.7);
    eff_axis.legend(prop={'size': 17, 'weight': 'bold'}, loc='lower right')

    axes_array[1, 2].axis('off')
    plt.tight_layout(pad=3.0)
    plt.savefig('universal_parity_plot_compressor_final.png', dpi=600, bbox_inches='tight')
    print("\n[✔] 压缩机训练集学术验证图已生成。")





SYSTEM_DESIGN_POINT = {'power_w': 1, 'Back_Pressure_bar': 1}

# 1. 动态获取当前运行环境的根目录路径（兼容标准脚本与 Jupyter 环境）
try:
    current_run_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_run_dir = os.getcwd()  # 如果 __file__ 不存在，则获取当前 Notebook 的工作目录

# 2. 锁定压缩机插致密化数据集的绝对路径
dataset_dir = os.path.join(current_run_dir, "Interpolated_Datasets")

# 3. 打印路径用于机理校验
print(f"正在锁定压缩机真实数据目录: {dataset_dir}")

# 4. 检索目标文件夹下所有有效的物理工况 CSV 文件
all_csv_files = os.listdir(dataset_dir)
all_csv_absolute_paths = [os.path.join(dataset_dir, f) for f in all_csv_files if f.endswith(".csv") and not f.startswith("._")]
all_csv_absolute_paths.sort()
master_data = load_and_unify_all_datasets(all_csv_absolute_paths)
master_data = keep_high_eta_per_power(master_data)
X_uni, Y_uni = create_universal_features_and_targets(master_data, SYSTEM_DESIGN_POINT)

X_train, X_test, Y_train, Y_test = train_test_split(X_uni, Y_uni, test_size=0.3, random_state=42)
scaler_X = PhysicsAwareScaler()
scaler_Y = PassThroughScaler()  # 压缩机原生 Y 无需拉伸

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
Y_train_scaled = scaler_Y.fit_transform(Y_train)

scaler_X = PhysicsAwareScaler()
scaler_Y = PassThroughScaler()  # 压缩机原生 Y 无需拉伸

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
Y_train_scaled = scaler_Y.fit_transform(Y_train)

optimized_inverse_mlp =  train_mlp_with_gaussian_hard_mining(
            X_train_scaled=X_train_scaled, Y_train_scaled=Y_train_scaled,
            scaler_X=scaler_X, scaler_Y=scaler_Y, design_constants=SYSTEM_DESIGN_POINT, duplication_factor=5
        )
calculate_and_visualize_model_accuracy_strict_physics(optimized_inverse_mlp, X_test, X_test_scaled, Y_test,scaler_Y)

# ==========================================
# 规范化模型与特征缩放器持久化存储（回归初始版本逻辑）
# ==========================================
print("\n--- 正在保存模型与特征缩放器 ---")

# 1. 动态获取当前运行环境的根目录（完美兼容 Jupyter 与标准脚本）
try:
    base_run_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_run_dir = os.getcwd()

# 2. 锁定并创建模型存放的绝对路径文件夹
model_save_dir = os.path.join(base_run_dir, 'saved_models')
os.makedirs(model_save_dir, exist_ok=True)

# 3. 构建清晰、自解释的持久化文件路径变量
entire_model_save_path = os.path.join(model_save_dir, 'compressor_inverse_mlp.pkl')
scaler_x_save_path = os.path.join(model_save_dir, 'scaler_X.pkl')
scaler_y_save_path = os.path.join(model_save_dir, 'scaler_Y.pkl')

# 4. 执行全量对象序列化（完全对齐最初版本的全模型打包文件）
joblib.dump(optimized_inverse_mlp, entire_model_save_path)
joblib.dump(scaler_X, scaler_x_save_path)
joblib.dump(scaler_Y, scaler_y_save_path)

# 如果你仍想保留纯 PyTorch 权重文件作为备份，可以解除下面这行代码的注释：
# torch.save(optimized_inverse_mlp.model.state_dict(), os.path.join(model_save_dir, 'compressor_weights.pth'))

print(f"[✔] 全量模型包装类与 Scaler 已成功保存至 '{model_save_dir}' 文件夹！")