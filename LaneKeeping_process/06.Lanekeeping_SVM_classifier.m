%% GC feature SVM
windows = 5;
channel = 4;
trial = 118;

GC_feature_X = zeros(channel, channel, windows, trial);
GC_feature_y = zeros(windows, trial);
GC_feature_trial = zeros(windows, trial);

trial_label_index = readtable("I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\" + ...
    "Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\LK_G1G2_RT_label.csv");

csvdatapath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\MVGC_csvdata\";

for i=1:height(trial_label_index)
    for w=1:5
        tN = trial_label_index.trialNumber(i);
        csvname = strcat("01_", num2str(tN, "%03d"), "_", num2str(w, "%02d"), ".csv");
        GC_feature_X(:, :, w, i) = readmatrix(strcat(csvdatapath, csvname));
        GC_feature_y(w, i) = trial_label_index.label(i);
        GC_feature_trial(w, i) = trial_label_index.trialNumber(i);
    end
end

GC_feature_X_f = reshape(GC_feature_X, channel, channel, windows*trial);
GC_feature_X_f = reshape(GC_feature_X_f, channel*channel, windows*trial);
GC_feature_X_f = GC_feature_X_f';
GC_feature_X_f(GC_feature_X_f == 0) = NaN;
GC_feature_X_f = GC_feature_X_f(~isnan(GC_feature_X_f));
GC_feature_X_f = reshape(GC_feature_X_f, windows*trial, (channel*channel)-4);

GC_feature_y_f = reshape(GC_feature_y, 1, numel(GC_feature_y));
GC_feature_y_f = GC_feature_y_f';

GC_feature_trial_f = reshape(GC_feature_trial, 1, numel(GC_feature_trial));
GC_feature_trial_f = GC_feature_trial_f';

errorRates = zeros(10, 1);
accRates = zeros(10, 10);

for i = 1:10
    permutedIndices = randperm(size(GC_feature_X_f, 1));
    val_size = floor(0.2 * length(GC_feature_y_f));
    val_indices = permutedIndices(1:val_size);
    train_indices = permutedIndices(val_size+1:end);
    
    XTrain = GC_feature_X_f(train_indices, :);
    yTrain = GC_feature_y_f(train_indices);
    XVal = GC_feature_X_f(val_indices, :);
    yVal = GC_feature_y_f(val_indices);
    % train SVM
    SVMModel = fitcsvm(XTrain, yTrain);

    % 10-fold cross validation
    CVSVMModel = crossval(SVMModel, 'KFold', 10);
    
    % Calculate the error rate for each cross-validation
    errorRates(i) = kfoldLoss(CVSVMModel);
    for j=1:10
        [~,score] = predict(CVSVMModel.Trained{j}, XVal);
        [~,pred] = max(score, [], 2);
        accRates(i, j) = sum(yVal == pred-1) / length(yVal);
    end
end

mean_acc = mean(accRates, "all");
std_acc = std(accRates, 0, "all");

%% TCDF feature SVM
windows = 5;
channel = 4;
trial = 118;

TCDF_feature_X = zeros(channel, channel, windows, trial);
TCDF_feature_y = zeros(windows, trial);

trial_label_index = readtable("I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\" + ...
    "Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\LK_G1G2_RT_label.csv");

csvdatapath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\" + ...
    "data\Lanekeeping_RT_connectivity_feature\TCDF_csvdata\";

for i=1:height(trial_label_index)
    for w=1:5
        tN = trial_label_index.trialNumber(i);
        csvname = strcat("01_", num2str(tN, "%03d"), "_", num2str(w, "%02d"), ".csv");
        TCDF_feature_X(:, :, w, i) = readmatrix(strcat(csvdatapath, csvname));
        TCDF_feature_y(w, i) = trial_label_index.label(i);
    end
end

TCDF_feature_X_f = reshape(TCDF_feature_X, channel, channel, windows*trial);
TCDF_feature_X_f = reshape(TCDF_feature_X_f, channel*channel, windows*trial);
TCDF_feature_X_f = TCDF_feature_X_f';

TCDF_feature_y_f = reshape(TCDF_feature_y, 1, numel(TCDF_feature_y));
TCDF_feature_y_f = TCDF_feature_y_f';

%% TCDF 沒有對角線
windows = 5;
channel = 4;
trial = 118;

TCDF_feature_X = zeros(channel, channel, windows, trial);
TCDF_feature_y = zeros(windows, trial);

trial_label_index = readtable("I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\" + ...
    "Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\LK_G1G2_RT_label.csv");

csvdatapath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\TCDF_csvdata\";

for i=1:height(trial_label_index)
    for w=1:5
        tN = trial_label_index.trialNumber(i);
        csvname = strcat("01_", num2str(tN, "%03d"), "_", num2str(w, "%02d"), ".csv");
        data = readmatrix(strcat(csvdatapath, csvname));
        data(logical(eye(channel))) = 0;
        TCDF_feature_X(:, :, w, i) = data;
        TCDF_feature_y(w, i) = trial_label_index.label(i);
    end
end

TCDF_feature_X_f = reshape(TCDF_feature_X, channel, channel, windows*trial);
TCDF_feature_X_f = reshape(TCDF_feature_X_f, channel*channel, windows*trial);
TCDF_feature_X_f = TCDF_feature_X_f';
TCDF_feature_X_f(TCDF_feature_X_f == 0) = NaN;
TCDF_feature_X_f = TCDF_feature_X_f(~isnan(TCDF_feature_X_f));
TCDF_feature_X_f = reshape(TCDF_feature_X_f, windows*trial, (channel*channel)-4);

TCDF_feature_y_f = reshape(TCDF_feature_y, 1, numel(TCDF_feature_y));
TCDF_feature_y_f = TCDF_feature_y_f';

errorRates = zeros(10, 1);
accRates = zeros(10, 10);

for i = 1:10
    permutedIndices = randperm(size(TCDF_feature_X_f, 1));
    val_size = floor(0.2 * length(TCDF_feature_y_f));
    val_indices = permutedIndices(1:val_size);
    train_indices = permutedIndices(val_size+1:end);
    
    XTrain = TCDF_feature_X_f(train_indices, :);
    yTrain = TCDF_feature_y_f(train_indices);
    XVal = TCDF_feature_X_f(val_indices, :);
    yVal = TCDF_feature_y_f(val_indices);
    % train SVM
    SVMModel = fitcsvm(XTrain, yTrain);

    % 10-fold cross validation
    CVSVMModel = crossval(SVMModel, 'KFold', 10);
    
    % Calculate the error rate for each cross-validation
    errorRates(i) = kfoldLoss(CVSVMModel);
    for j=1:10
        [~,score] = predict(CVSVMModel.Trained{j}, XVal);
        [~,pred] = max(score, [], 2);
        accRates(i, j) = sum(yVal == pred-1) / length(yVal);
    end
end

mean_acc = mean(accRates, "all");
std_acc = std(accRates, 0, "all");

%% power feature SVM
windows = 5;
channel = 4;
trial = 118;

power_feature_X = zeros(channel, channel, windows, trial);
power_feature_y = zeros(windows, trial);

trial_label_index = readtable("I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\" + ...
    "Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\LK_G1G2_RT_label.csv");

csvdatapath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\power_csvdata\";

for i=1:height(trial_label_index)
    for w=1:5
        tN = trial_label_index.trialNumber(i);
        csvname = strcat("01_", num2str(tN, "%03d"), "_", num2str(w, "%02d"), ".csv");
        power_feature_X(:, :, w, i) = rescale(readmatrix(strcat(csvdatapath, csvname)), 0, 1);
        power_feature_y(w, i) = trial_label_index.label(i);
    end
end

power_feature_X_f = reshape(power_feature_X, channel, channel, windows*trial);
power_feature_X_f = reshape(power_feature_X_f, channel*channel, windows*trial);
power_feature_X_f = power_feature_X_f';

power_feature_y_f = reshape(power_feature_y, 1, numel(power_feature_y));
power_feature_y_f = power_feature_y_f';

errorRates = zeros(10, 1);
accRates = zeros(100, 10);

for i = 1:100
    permutedIndices = randperm(size(power_feature_X_f, 1));
    val_size = floor(0.2 * length(power_feature_y_f));
    val_indices = permutedIndices(1:val_size);
    train_indices = permutedIndices(val_size+1:end);
    
    XTrain = power_feature_X_f(train_indices, :);
    yTrain = power_feature_y_f(train_indices);
    XVal = power_feature_X_f(val_indices, :);
    yVal = power_feature_y_f(val_indices);
    % train SVM
    SVMModel = fitcsvm(XTrain, yTrain);

    % 10-fold cross validation
    CVSVMModel = crossval(SVMModel, 'KFold', 10);
    
    % Calculate the error rate for each cross-validation
    errorRates(i) = kfoldLoss(CVSVMModel);
    for j=1:10
        [~,score] = predict(CVSVMModel.Trained{j}, XVal);
        [~,pred] = max(score, [], 2);
        accRates(i, j) = sum(yVal == pred-1) / length(yVal);
    end
end

mean_acc = mean(mean(accRates, 2));
std_acc = std(accRates, 0, "all");

%% InfoFlowNet feature SVM
windows = 5;
channel = 4;
trial = 118;

Info_feature_X = zeros(channel, channel, windows, trial);
Info_feature_y = zeros(windows, trial);

trial_label_index = readtable("I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\" + ...
    "Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\LK_G1G2_RT_label.csv");

csvdatapath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\" + ...
    "data\Lanekeeping_RT_connectivity_feature\head2_eye_csvdata\";

for i=1:height(trial_label_index)
    for w=1:5
        tN = trial_label_index.trialNumber(i);
        csvname = strcat("01_", num2str(tN, "%03d"), "_", num2str(w, "%02d"), ".csv");
        Info_feature_X(:, :, w, i) = readmatrix(strcat(csvdatapath, csvname));
        Info_feature_y(w, i) = trial_label_index.label(i);
    end
end

Info_feature_X_f = reshape(Info_feature_X, channel, channel, windows*trial);
Info_feature_X_f = reshape(Info_feature_X_f, channel*channel, windows*trial);
Info_feature_X_f = Info_feature_X_f';

Info_feature_y_f = reshape(Info_feature_y, 1, numel(Info_feature_y));
Info_feature_y_f = Info_feature_y_f';

%% InfoFlowNet 沒有對角線
windows = 5;
channel = 4;
trial = 118;

Info_feature_X = zeros(channel, channel, windows, trial);
Info_feature_y = zeros(windows, trial);

trial_label_index = readtable("I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\" + ...
    "Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\LK_G1G2_RT_label.csv");

csvdatapath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\" + ...
    "data\Lanekeeping_RT_connectivity_feature\head4_csvdata\";

for i=1:height(trial_label_index)
    for w=1:5
        tN = trial_label_index.trialNumber(i);
        csvname = strcat("01_", num2str(tN, "%03d"), "_", num2str(w, "%02d"), ".csv");
        data = readmatrix(strcat(csvdatapath, csvname));
        data(logical(eye(channel))) = nan;
        Info_feature_X(:, :, w, i) = data;
        Info_feature_y(w, i) = trial_label_index.label(i);
    end
end

Info_feature_X_f = reshape(Info_feature_X, channel, channel, windows*trial);
Info_feature_X_f = reshape(Info_feature_X_f, channel*channel, windows*trial);
Info_feature_X_f = Info_feature_X_f';
Info_feature_X_f = Info_feature_X_f(~isnan(Info_feature_X_f));
Info_feature_X_f = reshape(Info_feature_X_f, windows*trial, (channel*channel)-4);

Info_feature_y_f = reshape(Info_feature_y, 1, numel(Info_feature_y));
Info_feature_y_f = Info_feature_y_f';

errorRates = zeros(10, 1);
accRates = zeros(100, 10);

for i = 1:100
    permutedIndices = randperm(size(Info_feature_X_f, 1));
    val_size = floor(0.2 * length(Info_feature_y_f));
    val_indices = permutedIndices(1:val_size);
    train_indices = permutedIndices(val_size+1:end);
    
    XTrain = Info_feature_X_f(train_indices, :);
    yTrain = Info_feature_y_f(train_indices);
    XVal = Info_feature_X_f(val_indices, :);
    yVal = Info_feature_y_f(val_indices);
    % train SVM
    SVMModel = fitcsvm(XTrain, yTrain);

    % 10-fold cross validation
    CVSVMModel = crossval(SVMModel, 'KFold', 10);
    
    % Calculate the error rate for each cross-validation
    errorRates(i) = kfoldLoss(CVSVMModel);
    for j=1:10
        [~,score] = predict(CVSVMModel.Trained{j}, XVal);
        [~,pred] = max(score, [], 2);
        accRates(i, j) = sum(yVal == pred-1) / length(yVal);
    end
end

mean_acc = mean(mean(accRates));
std_acc = mean(std(accRates, 0, 1));

%% GC + power feature SVM

powerGC_feature_X_f = [GC_feature_X_f power_feature_X_f];
powerGC_feature_y_f = power_feature_y_f;

errorRates = zeros(10, 1);
accRates = zeros(10, 10);

for i = 1:10
    permutedIndices = randperm(size(powerGC_feature_X_f, 1));
    val_size = floor(0.2 * length(powerGC_feature_y_f));
    val_indices = permutedIndices(1:val_size);
    train_indices = permutedIndices(val_size+1:end);
    
    XTrain = powerGC_feature_X_f(train_indices, :);
    yTrain = powerGC_feature_y_f(train_indices);
    XVal = powerGC_feature_X_f(val_indices, :);
    yVal = powerGC_feature_y_f(val_indices);
    % train SVM
    SVMModel = fitcsvm(XTrain, yTrain);

    % 10-fold cross validation
    CVSVMModel = crossval(SVMModel, 'KFold', 10);
    
    % Calculate the error rate for each cross-validation
    errorRates(i) = kfoldLoss(CVSVMModel);
    for j=1:10
        [~,score] = predict(CVSVMModel.Trained{j}, XVal);
        [~,pred] = max(score, [], 2);
        accRates(i, j) = sum(yVal == pred-1) / length(yVal);
    end
end

mean_acc = mean(accRates, "all");
std_acc = std(accRates, 0, "all");

%% TCDF + power feature SVM

powerTCDF_feature_X_f = [TCDF_feature_X_f power_feature_X_f];
powerTCDF_feature_y_f = power_feature_y_f;

errorRates = zeros(10, 1);
accRates = zeros(10, 10);

for i = 1:10
    permutedIndices = randperm(size(powerTCDF_feature_X_f, 1));
    val_size = floor(0.2 * length(powerTCDF_feature_y_f));
    val_indices = permutedIndices(1:val_size);
    train_indices = permutedIndices(val_size+1:end);
    
    XTrain = powerTCDF_feature_X_f(train_indices, :);
    yTrain = powerTCDF_feature_y_f(train_indices);
    XVal = powerTCDF_feature_X_f(val_indices, :);
    yVal = powerTCDF_feature_y_f(val_indices);
    % train SVM
    SVMModel = fitcsvm(XTrain, yTrain);

    % 10-fold cross validation
    CVSVMModel = crossval(SVMModel, 'KFold', 10);
    
    % Calculate the error rate for each cross-validation
    errorRates(i) = kfoldLoss(CVSVMModel);
    for j=1:10
        [~,score] = predict(CVSVMModel.Trained{j}, XVal);
        [~,pred] = max(score, [], 2);
        accRates(i, j) = sum(yVal == pred-1) / length(yVal);
    end
end

mean_acc = mean(accRates, "all");
std_acc = std(accRates, 0, "all");

%% InfoFlowNet + power feature SVM

powerInfo_feature_X_f = [Info_feature_X_f power_feature_X_f];
powerInfo_feature_y_f = power_feature_y_f;

errorRates = zeros(10, 1);
accRates = zeros(100, 10);

for i = 1:100
    permutedIndices = randperm(size(powerInfo_feature_X_f, 1));
    val_size = floor(0.2 * length(powerInfo_feature_y_f));
    val_indices = permutedIndices(1:val_size);
    train_indices = permutedIndices(val_size+1:end);
    
    XTrain = powerInfo_feature_X_f(train_indices, :);
    yTrain = powerInfo_feature_y_f(train_indices);
    XVal = powerInfo_feature_X_f(val_indices, :);
    yVal = powerInfo_feature_y_f(val_indices);
    % train SVM
    SVMModel = fitcsvm(XTrain, yTrain);

    % 10-fold cross validation
    CVSVMModel = crossval(SVMModel, 'KFold', 10);
    
    % Calculate the error rate for each cross-validation
    errorRates(i) = kfoldLoss(CVSVMModel);
    for j=1:10
        [~,score] = predict(CVSVMModel.Trained{j}, XVal);
        [~,pred] = max(score, [], 2);
        accRates(i, j) = sum(yVal == pred-1) / length(yVal);
    end
end

mean_acc = mean(accRates, "all");
std_acc = std(accRates, 0, "all");

%% GC + TCDF + IFNet + power

GC_accRates = zeros(100, 10);
TCDF_accRates = zeros(100, 10);
Info_accRates = zeros(100, 10);
power_accRates = zeros(100, 10);
powerGC_accRates = zeros(100, 10);
powerTCDF_accRates = zeros(100, 10);
powerInfo_accRates = zeros(100, 10);

powerGC_feature_X_f = [GC_feature_X_f power_feature_X_f];
powerGC_feature_y_f = power_feature_y_f;

powerTCDF_feature_X_f = [TCDF_feature_X_f power_feature_X_f];
powerTCDF_feature_y_f = power_feature_y_f;

powerInfo_feature_X_f = [Info_feature_X_f power_feature_X_f];
powerInfo_feature_y_f = power_feature_y_f;


for i=1:100
    val_size = floor(0.2 * length(GC_feature_y));
    val_indices = permutedIndices(i, 1:val_size);
    train_indices = permutedIndices(i, val_size+1:end);
    
    %% GC
    XTrain = GC_feature_X_f(train_indices, :);
    yTrain = GC_feature_y_f(train_indices);
    XVal = GC_feature_X_f(val_indices, :);
    yVal = GC_feature_y_f(val_indices);
    % train SVM
    SVMModel = fitcsvm(XTrain, yTrain);
    
    % 10-fold cross validation
    CVSVMModel = crossval(SVMModel, 'KFold', 10);
    
    % Calculate the error rate for each cross-validation
    GC_errorRates = kfoldLoss(CVSVMModel);
    for j=1:10
        [~,score] = predict(CVSVMModel.Trained{j}, XVal);
        [~,pred] = max(score, [], 2);
        GC_accRates(i, j) = sum(yVal == pred-1) / length(yVal);
    end
    
    %% TCDF
    XTrain = TCDF_feature_X_f(train_indices, :);
    yTrain = TCDF_feature_y_f(train_indices);
    XVal = TCDF_feature_X_f(val_indices, :);
    yVal = TCDF_feature_y_f(val_indices);
    % train SVM
    SVMModel = fitcsvm(XTrain, yTrain);
    
    % 10-fold cross validation
    CVSVMModel = crossval(SVMModel, 'KFold', 10);
    
    % Calculate the error rate for each cross-validation
    TCDF_errorRates = kfoldLoss(CVSVMModel);
    for j=1:10
        [~,score] = predict(CVSVMModel.Trained{j}, XVal);
        [~,pred] = max(score, [], 2);
        TCDF_accRates(i, j) = sum(yVal == pred-1) / length(yVal);
    end
    
    %% Info
    XTrain = Info_feature_X_f(train_indices, :);
    yTrain = Info_feature_y_f(train_indices);
    XVal = Info_feature_X_f(val_indices, :);
    yVal = Info_feature_y_f(val_indices);
    % train SVM
    SVMModel = fitcsvm(XTrain, yTrain);
    
    % 10-fold cross validation
    CVSVMModel = crossval(SVMModel, 'KFold', 10);
    
    % Calculate the error rate for each cross-validation
    Info_errorRates = kfoldLoss(CVSVMModel);
    for j=1:10
        [~,score] = predict(CVSVMModel.Trained{j}, XVal);
        [~,pred] = max(score, [], 2);
        Info_accRates(i, j) = sum(yVal == pred-1) / length(yVal);
    end
    
    %% power
    XTrain = power_feature_X_f(train_indices, :);
    yTrain = power_feature_y_f(train_indices);
    XVal = power_feature_X_f(val_indices, :);
    yVal = power_feature_y_f(val_indices);
    % train SVM
    SVMModel = fitcsvm(XTrain, yTrain);
    
    % 10-fold cross validation
    CVSVMModel = crossval(SVMModel, 'KFold', 10);
    
    % Calculate the error rate for each cross-validation
    power_errorRates = kfoldLoss(CVSVMModel);
    for j=1:10
        [~,score] = predict(CVSVMModel.Trained{j}, XVal);
        [~,pred] = max(score, [], 2);
        power_accRates(i, j) = sum(yVal == pred-1) / length(yVal);
    end
    
    %% power + GC
    XTrain = powerGC_feature_X_f(train_indices, :);
    yTrain = powerGC_feature_y_f(train_indices);
    XVal = powerGC_feature_X_f(val_indices, :);
    yVal = powerGC_feature_y_f(val_indices);
    % train SVM
    SVMModel = fitcsvm(XTrain, yTrain);
    
    % 10-fold cross validation
    CVSVMModel = crossval(SVMModel, 'KFold', 10);
    
    % Calculate the error rate for each cross-validation
    powerGC_errorRates = kfoldLoss(CVSVMModel);
    for j=1:10
        [~,score] = predict(CVSVMModel.Trained{j}, XVal);
        [~,pred] = max(score, [], 2);
        powerGC_accRates(i, j) = sum(yVal == pred-1) / length(yVal);
    end
    
    %% power + TCDF
    XTrain = powerTCDF_feature_X_f(train_indices, :);
    yTrain = powerTCDF_feature_y_f(train_indices);
    XVal = powerTCDF_feature_X_f(val_indices, :);
    yVal = powerTCDF_feature_y_f(val_indices);
    % train SVM
    SVMModel = fitcsvm(XTrain, yTrain);
    
    % 10-fold cross validation
    CVSVMModel = crossval(SVMModel, 'KFold', 10);
    
    % Calculate the error rate for each cross-validation
    powerTCDF_errorRates = kfoldLoss(CVSVMModel);
    for j=1:10
        [~,score] = predict(CVSVMModel.Trained{j}, XVal);
        [~,pred] = max(score, [], 2);
        powerTCDF_accRates(i, j) = sum(yVal == pred-1) / length(yVal);
    end
    
    %% power + Info
    XTrain = powerInfo_feature_X_f(train_indices, :);
    yTrain = powerInfo_feature_y_f(train_indices);
    XVal = powerInfo_feature_X_f(val_indices, :);
    yVal = powerInfo_feature_y_f(val_indices);
    % train SVM
    SVMModel = fitcsvm(XTrain, yTrain);
    
    % 10-fold cross validation
    CVSVMModel = crossval(SVMModel, 'KFold', 10);
    
    % Calculate the error rate for each cross-validation
    powerInfo_errorRates = kfoldLoss(CVSVMModel);
    for j=1:10
        [~,score] = predict(CVSVMModel.Trained{j}, XVal);
        [~,pred] = max(score, [], 2);
        powerInfo_accRates(i, j) = sum(yVal == pred-1) / length(yVal);
    end
end

GC_res(1) = mean(GC_accRates(:));
GC_res(2) = std(GC_accRates(:), 0, 1);

TCDF_res(1) = mean(TCDF_accRates(:));
TCDF_res(2) = std(TCDF_accRates(:), 0, 1);

Info_res(1) = mean(Info_accRates(:));
Info_res(2) = std(Info_accRates(:), 0, 1);

power_res(1) = mean(power_accRates(:));
power_res(2) = std(power_accRates(:), 0, 1);

powerGC_res(1) = mean(powerGC_accRates(:));
powerGC_res(2) = std(powerGC_accRates(:), 0, 1);

powerTCDF_res(1) = mean(powerTCDF_accRates(:));
powerTCDF_res(2) = std(powerTCDF_accRates(:), 0, 1);

powerInfo_res(1) = mean(powerInfo_accRates(:));
powerInfo_res(2) = std(powerInfo_accRates(:), 0, 1);