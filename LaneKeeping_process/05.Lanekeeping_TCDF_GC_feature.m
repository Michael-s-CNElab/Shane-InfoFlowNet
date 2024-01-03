%% TCDF connectivity graph
windows = 2;
channel = 4;
trial = 581;

ch_name = {'Fz' 'Cz' 'Pz' 'Oz'};

att_score = zeros(channel, channel, windows, trial);

csvPath = 'I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\data\LanekeepingforTCDF\result\';
TCDF_result = dir([csvPath, '*.csv']);

for f=1:6972
    if contains(TCDF_result(f).name, 'Attention_Scores.csv')
        if contains(TCDF_result(f).name, 'baseline_01')
            strs = strsplit(TCDF_result(f).name, '_');
            tr = str2num(strs{2});
            att_score(:, :, 1, tr) = csvread(strcat(csvPath, TCDF_result(f).name));
        end
        if contains(TCDF_result(f).name, 'RT')
            strs = strsplit(TCDF_result(f).name, '_');
            tr = str2num(strs{2});
            att_score(:, :, 2, tr) = csvread(strcat(csvPath, TCDF_result(f).name));
        end
    end
end

H_array = nan(channel, channel);

for c_0=1:channel
    for c=1:channel
        a = squeeze(att_score(c_0, c, 1, :));
        b = squeeze(att_score(c_0, c, 2, :));

        H_array(c_0, c) = ttest(a, b);
    end
end

RT_window_feature = mean(att_score(:, :, 2, :), 4);

weight = transpose(RT_window_feature);
node_names = {'Fz','Cz','Pz', 'Oz'};

GG = digraph(weight, node_names);
hh = plot(GG);
hh.LineWidth = 3;
hh.EdgeCData = GG.Edges.Weight;
hh.EdgeFontSize = 20;
hh.NodeFontSize = 24;
hh.NodeColor = 'r';
hh.MarkerSize = 16;
hh.XData = [-2 2 -2 2];
hh.YData = [2 2 -2 -2];
hh.NodeLabelMode = 'auto';
hh.ArrowSize = 24;
hh.EdgeAlpha = 1;
hh.ArrowPosition = 0.9;
set(gca, 'Color', 'none', 'box', 'off', 'Visible', 'off');
set(gcf,'position',[100,100,800,800]);
caxis([0 1.0]);
colorbar('FontSize', 18, 'position',[0.9,0.3,0.04,0.5]);
box off

%% TCDF connectivity feature

windows = 5;
channel = 4;
trial = 581;
subject = 1;

ch_name = {'Fz' 'Cz' 'Pz' 'Oz'};

att_score = zeros(channel, channel, windows, trial);

csvPath = 'I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\data\LanekeepingforTCDF\result\';
TCDF_result = dir([csvPath, '*.csv']);

for f=1:6972
    if contains(TCDF_result(f).name, 'Attention_Scores.csv')
        if contains(TCDF_result(f).name, 'baseline_01')
            strs = strsplit(TCDF_result(f).name, '_');
            tr = str2num(strs{2});
            att_score(:, :, 1, tr) = csvread(strcat(csvPath, TCDF_result(f).name));
        end
        if contains(TCDF_result(f).name, 'baseline_02')
            strs = strsplit(TCDF_result(f).name, '_');
            tr = str2num(strs{2});
            att_score(:, :, 2, tr) = csvread(strcat(csvPath, TCDF_result(f).name));
        end
        if contains(TCDF_result(f).name, 'baseline_03')
            strs = strsplit(TCDF_result(f).name, '_');
            tr = str2num(strs{2});
            att_score(:, :, 3, tr) = csvread(strcat(csvPath, TCDF_result(f).name));
        end
        if contains(TCDF_result(f).name, 'baseline_04')
            strs = strsplit(TCDF_result(f).name, '_');
            tr = str2num(strs{2});
            att_score(:, :, 4, tr) = csvread(strcat(csvPath, TCDF_result(f).name));
        end
        if contains(TCDF_result(f).name, 'baseline_05')
            strs = strsplit(TCDF_result(f).name, '_');
            tr = str2num(strs{2});
            att_score(:, :, 5, tr) = csvread(strcat(csvPath, TCDF_result(f).name));
        end
    end
end

trial_label_index = readtable("I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\" + ...
    "Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\LK_G1G2_RT_label.csv");
savePath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\TCDF_csvdata\";

for i=1:height(trial_label_index)
    for w=1:5
        feature = att_score(:, :, w, trial_label_index.trialNumber(i));
        csvname = strcat(savePath, num2str(subject, "%02d"), "_", ...
            num2str(trial_label_index.trialNumber(i), "%03d"), "_", ...
            num2str(w, "%02d"), ".csv");
        csvwrite(csvname, feature);
    end
end

%% TCDF 8ch connectivity feature

windows = 5;
channel = 8;
trial = 581;
subject = 1;

ch_name = {'Fz' 'F3' 'F4' 'Cz' 'C3' 'C4' 'Pz' 'Oz'};

att_score = zeros(channel, channel, windows, trial);

for f=1:6972
    if contains(TCDF_result(f).name, 'Attention_Scores.csv')
        if contains(TCDF_result(f).name, 'baseline_01')
            strs = strsplit(TCDF_result(f).name, '_');
            tr = str2num(strs{2});
            att_score(:, :, 1, tr) = csvread(strcat(csvPath, TCDF_result(f).name));
        end
        if contains(TCDF_result(f).name, 'baseline_02')
            strs = strsplit(TCDF_result(f).name, '_');
            tr = str2num(strs{2});
            att_score(:, :, 2, tr) = csvread(strcat(csvPath, TCDF_result(f).name));
        end
        if contains(TCDF_result(f).name, 'baseline_03')
            strs = strsplit(TCDF_result(f).name, '_');
            tr = str2num(strs{2});
            att_score(:, :, 3, tr) = csvread(strcat(csvPath, TCDF_result(f).name));
        end
        if contains(TCDF_result(f).name, 'baseline_04')
            strs = strsplit(TCDF_result(f).name, '_');
            tr = str2num(strs{2});
            att_score(:, :, 4, tr) = csvread(strcat(csvPath, TCDF_result(f).name));
        end
        if contains(TCDF_result(f).name, 'baseline_05')
            strs = strsplit(TCDF_result(f).name, '_');
            tr = str2num(strs{2});
            att_score(:, :, 5, tr) = csvread(strcat(csvPath, TCDF_result(f).name));
        end
    end
end

trial_label_index = readtable("I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\" + ...
    "Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\LK_G1G2_RT_label.csv");
savePath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\" + ...
    "data\Lanekeeping_RT_connectivity_feature\TCDF_8ch_csvdata\";

for i=1:height(trial_label_index)
    for w=1:5
        feature = att_score(:, :, w, trial_label_index.trialNumber(i));
        csvname = strcat(savePath, num2str(subject, "%02d"), "_", ...
            num2str(trial_label_index.trialNumber(i), "%03d"), "_", ...
            num2str(w, "%02d"), ".csv");
        csvwrite(csvname, feature);
    end
end

%%  GC connectivity feature

trial_label_index = readtable("I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\" + ...
    "Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\LK_G1G2_RT_label.csv");
windows = 5;
channel = 4;
trial = 118;
time_len = 100;
order = 29;

ch_name = {'Fz' 'Cz' 'Pz' 'Oz'};

csvdatapath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\data\LanekeepingforTCDFGC\csvdata\";
savePath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\MVGC_csvdata\";

for i=1:height(trial_label_index)
    for w=1:5
        tN = trial_label_index.trialNumber(i);
        csvname = strcat("01_", num2str(tN, "%03d"), "_baseline_", num2str(w, "%02d"), ".csv");
        data = transpose(readmatrix(strcat(csvdatapath, csvname)));
        res = MVGC_timedomain_GC(data, 1, order);
        res(isnan(res))=0;

        savecsvname = strcat(savePath, "01_", num2str(tN, "%03d"), "_", num2str(w, "%02d"), ".csv");
        writematrix(res, savecsvname)
    end
end

%%  GC 8ch connectivity feature

trial_label_index = readtable("I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\" + ...
    "Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\LK_G1G2_RT_label.csv");
windows = 5;
channel = 8;
trial = 118;
time_len = 100;
order = 1;

ch_name = {'Fz' 'F3' 'F4' 'Cz' 'C3' 'C4' 'Pz' 'Oz'};

csvdatapath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\data\Lanekeeping8chforTCDF\csvdata\";
savePath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\" + ...
    "data\Lanekeeping_RT_connectivity_feature\MVGC_8ch_csvdata\";

for i=1:height(trial_label_index)
    for w=1:5
        tN = trial_label_index.trialNumber(i);
        csvname = strcat("01_", num2str(tN, "%03d"), "_baseline_", num2str(w, "%02d"), ".csv");
        data = transpose(readmatrix(strcat(csvdatapath, csvname)));
        res = MVGC_timedomain_GC(data, 1, order);
        res(isnan(res))=0;
        savecsvname = strcat(savePath, "01_", num2str(tN, "%03d"), "_", num2str(w, "%02d"), ".csv");
        writematrix(res, savecsvname)
    end
end

%%  power feature

trial_label_index = readtable("I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\" + ...
    "Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\LK_G1G2_RT_label.csv");
windows = 5;
channel = 4;
trial = 118;
time_len = 100;
srate = 500;

ch_name = {'Fz' 'Cz' 'Pz' 'Oz'};

csvdatapath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\data\LanekeepingforTCDFGC\csvdata\";
savePath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\power_csvdata\";

for i=1:height(trial_label_index)
    for w=1:5
        tN = trial_label_index.trialNumber(i);
        csvname = strcat("01_", num2str(tN, "%03d"), "_baseline_", num2str(w, "%02d"), ".csv");
        data = readmatrix(strcat(csvdatapath, csvname));

        [tf, freqs, times, itcvals] = timefreq(data, srate, 'freqs', [1 50], 'nfreqs', 32, 'padratio', 32);
        
        power_feature = zeros(4, channel);
        for j=1:4
            power_feature(1, j) = abs(sum(mean(tf(1:3, :, j), 2), 1));
            power_feature(2, j) = abs(sum(mean(tf(4:6, :, j), 2), 1));
            power_feature(3, j) = abs(sum(mean(tf(7:9, :, j), 2), 1));
            power_feature(4, j) = abs(sum(mean(tf(10:20, :, j), 2), 1));
        end

        savecsvname = strcat(savePath, "01_", num2str(tN, "%03d"), "_", num2str(w, "%02d"), ".csv");
        writematrix(power_feature, savecsvname);
    end
end

%%  power 8ch feature

trial_label_index = readtable("I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\" + ...
    "Shane-InfoFlowNet\data\Lanekeeping_RT_connectivity_feature\LK_G1G2_RT_label.csv");
windows = 5;
channel = 8;
trial = 118;
time_len = 100;
srate = 500;

ch_name = {'Fz' 'F3' 'F4' 'Cz' 'C3' 'C4' 'Pz' 'Oz'};

csvdatapath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\data\Lanekeeping8chforTCDFGCPOWER\csvdata\";
savePath = "I:\共用雲端硬碟\CNElab_枋劭勳\10.交接資料\Shane-InfoFlowNet\" + ...
    "data\Lanekeeping_RT_connectivity_feature\power_8ch_csvdata\";

for i=1:height(trial_label_index)
    for w=1:5
        tN = trial_label_index.trialNumber(i);
        csvname = strcat("01_", num2str(tN, "%03d"), "_baseline_", num2str(w, "%02d"), ".csv");
        data = readmatrix(strcat(csvdatapath, csvname));

        [tf, freqs, times, itcvals] = timefreq(data, srate, 'freqs', [1 50], 'nfreqs', 32, 'padratio', 32);
        
        power_feature = zeros(4, channel);
        for j=1:channel
            power_feature(1, j) = abs(sum(mean(tf(1:3, :, j), 2), 1));
            power_feature(2, j) = abs(sum(mean(tf(4:6, :, j), 2), 1));
            power_feature(3, j) = abs(sum(mean(tf(7:9, :, j), 2), 1));
            power_feature(4, j) = abs(sum(mean(tf(10:20, :, j), 2), 1));
        end

        savecsvname = strcat(savePath, "01_", num2str(tN, "%03d"), "_", num2str(w, "%02d"), ".csv");
        writematrix(power_feature, savecsvname);
    end
end