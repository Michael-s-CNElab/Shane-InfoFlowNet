%% TCDF 8ch feature SVM
windows = 5;
channel = 8;
trial = 118;

TCDF_feature_X = zeros(channel, channel, windows, trial);
TCDF_feature_y = zeros(windows, trial);

trial_label_index = readtable("I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\" + ...
    "Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\LK_G1G2_RT_label.csv");

csvdatapath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\" + ...
    "data\Lanekeeping_RT_connectivity_feature\TCDF_8ch_csvdata\";

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

%% power 8ch feature SVM
windows = 5;
channel = 8;
trial = 118;

power_feature_X = zeros(4, channel, windows, trial);
power_feature_y = zeros(windows, trial);

trial_label_index = readtable("I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\" + ...
    "Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\LK_G1G2_RT_label.csv");

csvdatapath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\" + ...
    "data\Lanekeeping_RT_connectivity_feature\power_8ch_csvdata\";

for i=1:height(trial_label_index)
    for w=1:5
        tN = trial_label_index.trialNumber(i);
        csvname = strcat("01_", num2str(tN, "%03d"), "_", num2str(w, "%02d"), ".csv");
        power_feature_X(:, :, w, i) = readmatrix(strcat(csvdatapath, csvname));
        power_feature_y(w, i) = trial_label_index.label(i);
    end
end

power_feature_X_f = reshape(power_feature_X, 4, channel, windows*trial);
power_feature_X_f = reshape(power_feature_X_f, channel*4, windows*trial);
power_feature_X_f = power_feature_X_f';
power_feature_X_f = (power_feature_X_f - mean(power_feature_X_f, 2)) ./ std(power_feature_X_f, 0, 2);

power_feature_y_f = reshape(power_feature_y, 1, numel(power_feature_y));
power_feature_y_f = power_feature_y_f';


errorRates = zeros(10, 1);
accRates = zeros(10, 10);

for i = 1:10
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