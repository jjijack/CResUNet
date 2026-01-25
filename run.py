import torch
from torch.utils.data import DataLoader, random_split
from dataset import NCCorrectionDataset  # 导入新写的 Dataset
from models.baseline.CResU_Net import CRUNet
from config import experiment_params, model_params

def masked_mse_loss(pred, target, mask):
    """
    只计算有效区域(mask=1)的均方误差
    """
    diff = (pred - target) ** 2
    # 只保留有效部分的误差
    valid_diff = diff * mask 
    
    # 计算平均值：总误差 / 有效像素数
    # 加上 1e-6 防止除以 0
    loss = valid_diff.sum() / (mask.sum() + 1e-6)
    return loss

def run():
    # 1. 配置路径
    fc_path = './data/forecast_structured.nc'
    ra_path = './data/reanalysis_structured.nc'
    
    # 2. 初始化 Dataset
    # 它会自动扫描并在控制台打印出有多少个 Run 是时间匹配的
    full_dataset = NCCorrectionDataset(fc_path, ra_path)
    
    # 3. 划分训练/验证集
    total_size = len(full_dataset)
    if total_size == 0:
        print("错误：没有找到时间匹配的样本！请检查nc文件的时间单位是否一致。")
        return

    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    print(f"训练集: {len(train_ds)}, 验证集: {len(val_ds)}")

    # 4. DataLoader
    batch_size = model_params['CResU_Net']['batch_gen']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # 5. 模型初始化
    device = experiment_params['device']
    in_chan = model_params['CResU_Net']['core']['in_channels']
    out_chan = model_params['CResU_Net']['core']['out_channels']
    
    model = CRUNet(
        selected_dim=0, 
        in_channels=in_chan, 
        out_channels=out_chan, 
        device=device
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 使用自定义的 masked MSE 损失函数
    # criterion = torch.nn.MSELoss()

    # 6. 训练循环
    for epoch in range(10):
        model.train()
        total_loss = 0
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            # loss = criterion(pred, y)
            loss = masked_mse_loss(pred, y, mask)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, MSE Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    run()