csv_path = ['I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\Plot_attention&signal\' ...
    'Lanekeeping_head8_train_info.csv']
data = readmatrix(csv_path);

mean(data(2:101, 3))
std(data(2:101, 3), 0, 1)


csv_path = ['I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\Plot_attention&signal\' ...
    'Lanekeeping_head8_Record.csv']
data = readmatrix(csv_path);

data_p = data(:, 4:7) * 4;

mean(data_p(:, 2))
std(data_p(:, 2))

mean(data_p(:, 3))
std(data_p(:, 3))

mean(data_p(:, 4))
std(data_p(:, 4))