import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import os
import time
import h5py
import joblib  # 用于保存标准化器


# ==================== 数据加载和预处理（加入标准化） ====================
def load_and_preprocess_data(subsample_factor=1):
    """加载数据，返回原始曲线和结构"""
    print("正在加载数据...")
    velocity_structures = sio.loadmat("./km_vs_dz010.mat")['km_vs']
    velocity_curves = sio.loadmat("./Va_r.mat")['Va_r']
    print(f"原始数据形状 - 速度结构: {velocity_structures.shape}, 视速度曲线: {velocity_curves.shape}")
    print(f"数据量: {len(velocity_curves):,} 条")
    return velocity_curves, velocity_structures

# ==================== 数据集类 ====================
class VelocityDataset(Dataset):
    def __init__(self, velocity_curves, velocity_structures):
        self.velocity_curves = torch.FloatTensor(velocity_curves)
        self.velocity_structures = torch.FloatTensor(velocity_structures)

    def __len__(self):
        return len(self.velocity_curves)

    def __getitem__(self, idx):
        return self.velocity_curves[idx], self.velocity_structures[idx]


# ==================== 残差卷积块 ====================
class ResidualConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 8, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.silu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        w = self.se(out)
        out = out * w
        out = out + residual
        out = F.silu(out)
        return out


# ==================== 改进模型：保留深度维度 ====================
class DepthPreservingVelocityNet(nn.Module):
    def __init__(self, input_len=150, output_dim=100, dropout_rate=0.2):
        super().__init__()
        # 编码器：逐步下采样
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4, stride=2),  # 688 -> 344
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3, stride=2),  # 344 -> 172
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=2),  # 172 -> 86
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        # 多尺度残差块
        self.res_blocks = nn.Sequential(
            ResidualConvBlock(128, kernel_size=7, dilation=1, dropout=dropout_rate),
            ResidualConvBlock(128, kernel_size=7, dilation=2, dropout=dropout_rate),
            ResidualConvBlock(128, kernel_size=7, dilation=4, dropout=dropout_rate),
            ResidualConvBlock(128, kernel_size=7, dilation=8, dropout=dropout_rate),
        )

        # 解码器：上采样到原始长度
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # 86 -> 172
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # 172 -> 344
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),  # 344 -> 688
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv1d(16, 1, kernel_size=5, padding=2),
        )

        # 最终裁剪/插值到指定输出维度
        self.output_adapt = nn.AdaptiveAvgPool1d(output_dim)

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, input_len)
        x = x.unsqueeze(1)  # (B, 1, input_len)
        x = self.encoder(x)  # (B, 128, L_enc)
        x = self.res_blocks(x)  # (B, 128, L_enc)
        x = self.decoder(x)  # (B, 1, 688)
        x = self.output_adapt(x)  # (B, 1, output_dim)
        return x.squeeze(1)  # (B, output_dim)


# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, num_epochs=60, lr=0.001, device='cuda'):
    """训练模型，每10轮保存一次检查点"""
    model = model.to(device)
    checkpoint_dir = "model_checkpoints1"
    os.makedirs(checkpoint_dir, exist_ok=True)
    # criterion = nn.SmoothL1Loss(beta=1.0)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    train_losses = []
    val_losses = []
    learning_rates = []

    print(f"\n开始训练，共{num_epochs}轮...")
    print("=" * 50)

    for epoch in range(num_epochs):
        start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        for curves, structures in train_loader:
            curves, structures = curves.to(device), structures.to(device)
            # 轻微噪声增强
            curves = curves + 0.005 * torch.randn_like(curves)
            optimizer.zero_grad()
            outputs = model(curves)
            loss = criterion(outputs, structures)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for curves, structures in val_loader:
                curves, structures = curves.to(device), structures.to(device)
                outputs = model(curves)
                loss = criterion(outputs, structures)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f} | Time: {time.time() - start_time:.2f}s")

    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
    }, 'best_velocity_model.pth')

    print(f"\n训练完成！最终验证损失: {avg_val_loss:.6f}")
    return train_losses, val_losses, learning_rates


# ==================== 评估函数 ====================
def evaluate_model(model, test_loader, device='cuda'):
    """评估模型"""
    model.eval()
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for curves, structures in test_loader:
            curves = curves.to(device)
            outputs = model(curves)
            predictions.append(outputs.cpu().numpy())
            ground_truth.append(structures.numpy())

    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    # 计算评估指标
    mse = np.mean((predictions - ground_truth) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth))
    rmse = np.sqrt(mse)
    ss_res = np.sum((ground_truth - predictions) ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))

    # 相关系数
    corr_coeffs = [np.corrcoef(predictions[i], ground_truth[i])[0, 1] for i in range(len(predictions))]
    avg_corr = np.mean(corr_coeffs)
    explained_variance = 1 - np.var(predictions - ground_truth) / (np.var(ground_truth) + 1e-8)

    print("\n" + "=" * 50)
    print("模型评估结果:")
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²分数: {r2_score:.6f}")
    print(f"解释方差: {explained_variance:.6f}")
    print(f"平均相关系数: {avg_corr:.6f}")
    print("=" * 50)

    return predictions, ground_truth


# ==================== 可视化训练历史 ====================
def plot_training_history(train_losses, val_losses, learning_rates=None):
    fig = plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train loss', linewidth=2)
    plt.plot(val_losses, label='Val loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(train_losses, label='Train loss', linewidth=2)
    plt.plot(val_losses, label='Val loss', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.title('Loss (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if learning_rates:
        plt.subplot(1, 3, 3)
        plt.plot(learning_rates, 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.title('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_losses_to_mat(train_losses, val_losses, learning_rates, filename='loss_data.mat'):
    sio.savemat(filename, {
        'train_losses': np.array(train_losses).reshape(-1, 1),
        'val_losses': np.array(val_losses).reshape(-1, 1),
        'learning_rates': np.array(learning_rates).reshape(-1, 1),
        'epochs': np.arange(1, len(train_losses) + 1).reshape(-1, 1)
    })
    print(f"\n✓ 损失数据已保存到 {filename}")


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("视速度曲线反演 - 保留深度维度的模型")
    print("=" * 60)

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 1. 加载原始数据
    print("\n1. 加载原始数据...")
    velocity_curves, velocity_structures = load_and_preprocess_data(subsample_factor=1)

    # 2. 划分数据集
    print("\n2. 划分数据集...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        velocity_curves, velocity_structures, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    print(f"训练集: {X_train.shape[0]} | 验证集: {X_val.shape[0]} | 测试集: {X_test.shape[0]}")

    # 3. 创建 DataLoader
    batch_size = 64
    train_dataset = VelocityDataset(X_train, y_train)
    val_dataset = VelocityDataset(X_val, y_val)
    test_dataset = VelocityDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    # 4. 创建模型
    print("\n4. 创建模型...")
    model = DepthPreservingVelocityNet(input_len=velocity_curves.shape[1], output_dim=velocity_structures.shape[1])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")

    # 5. 训练
    print("\n5. 开始训练...")
    train_losses, val_losses, learning_rates = train_model(
        model, train_loader, val_loader,
        num_epochs=160, lr=0.0002, device=device
    )

    # 6. 保存损失曲线
    save_losses_to_mat(train_losses, val_losses, learning_rates)
    plot_training_history(train_losses, val_losses, learning_rates)

    # 7. 评估模型
    print("\n6. 评估模型...")
    best_model = DepthPreservingVelocityNet(input_len=velocity_curves.shape[1],
                                            output_dim=velocity_structures.shape[1]).to(device)
    checkpoint = torch.load('best_velocity_model.pth', map_location=device)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    predictions, ground_truth = evaluate_model(best_model, test_loader, device)

    print("\n" + "=" * 60)
    print("训练和评估完成！")
    print("已保存: best_velocity_model.pth, training_history.png, loss_data.mat")
    print("=" * 60)


# ==================== 快速预测函数 ====================
def quick_predict(input_file="./Va_r_all_models.mat", output_file="./predictions.mat",
                  checkpoint_path="./best_velocity_model.pth"):
    """对新数据做预测"""
    print("快速预测模式...")
    # 加载新数据
    new_curves = sio.loadmat(input_file)['Va_r']
    print(f"加载 {new_curves.shape[0]} 条视速度曲线")

    # 加载模型
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = DepthPreservingVelocityNet(input_len=new_curves.shape[1], output_dim=100).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 预测
    batch_size = 400
    predictions = []
    with torch.no_grad():
        for i in range(0, len(new_curves), batch_size):
            batch = torch.FloatTensor(new_curves[i:i + batch_size]).to(device)
            pred = model(batch).cpu().numpy()
            predictions.append(pred)
    predictions = np.vstack(predictions)

    sio.savemat(output_file, {'velocity_structures': predictions})
    print(f"✓ 预测完成，结果保存至 {output_file}")
    return predictions


if __name__ == "__main__":
    main()