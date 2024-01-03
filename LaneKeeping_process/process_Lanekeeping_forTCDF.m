Data = EEG.data;

s=1;
trial_info = {};
count = 1;
for t=1:683
    if length(EEG.epoch(t).eventlatency) >= 2
        if EEG.epoch(t).eventlatency{2} > 100
            trial_info{end+1, 1} = count;
            trial_info{end, 2} = s;
            trial_info{end, 3} = t;
            trial_info{end, 4} = EEG.epoch(t).eventlatency{2};
            count = count + 1;
        end
    end
end

RT_index = cell2mat(trial_info);
%% 4ch
data = ones(4, 1500, 581);

for t=1:581
    data(1, :, t) = EEG.data(5, :, RT_index(t, 3));  % Fz
    data(2, :, t) = EEG.data(15, :, RT_index(t, 3)); % Cz
    data(3, :, t) = EEG.data(25, :, RT_index(t, 3)); % Pz
    data(4, :, t) = EEG.data(31, :, RT_index(t, 3)); % Oz
end

data_name = 'LanekeepingforTCDF';
trial=581;
subject=1;
ch_name ={'Fz', 'Cz', 'Pz', 'Oz'};
save_path = strcat("I:\\共用雲端硬碟\\CNElab_枋劭勳\\10.交接資料\\Shane-InfoFlowNet\\data\\", data_name, "\\");

for s=1:subject
    for t=1:trial
        baseline_csv_name=strcat(num2str(s, "%02d"), '_', num2str(t, "%03d"), '_baseline_01.csv');
        baseline_data = transpose(data(:, 1:100, t));
        T = array2table(baseline_data);
        T.Properties.VariableNames(1:4) = ch_name;
        filepathname = strcat(save_path, "csvdata\\", baseline_csv_name);
        writetable(T, filepathname);

        baseline_csv_name=strcat(num2str(s, "%02d"), '_', num2str(t, "%03d"), '_baseline_02.csv');
        baseline_data = transpose(data(:, 101:200, t));
        T = array2table(baseline_data);
        T.Properties.VariableNames(1:4) = ch_name;
        filepathname = strcat(save_path, "csvdata\\", baseline_csv_name);
        writetable(T, filepathname);

        baseline_csv_name=strcat(num2str(s, "%02d"), '_', num2str(t, "%03d"), '_baseline_03.csv');
        baseline_data = transpose(data(:, 201:300, t));
        T = array2table(baseline_data);
        T.Properties.VariableNames(1:4) = ch_name;
        filepathname = strcat(save_path, "csvdata\\", baseline_csv_name);
        writetable(T, filepathname);

        baseline_csv_name=strcat(num2str(s, "%02d"), '_', num2str(t, "%03d"), '_baseline_04.csv');
        baseline_data = transpose(data(:, 301:400, t));
        T = array2table(baseline_data);
        T.Properties.VariableNames(1:4) = ch_name;
        filepathname = strcat(save_path, "csvdata\\", baseline_csv_name);
        writetable(T, filepathname);

        baseline_csv_name=strcat(num2str(s, "%02d"), '_', num2str(t, "%03d"), '_baseline_05.csv');
        baseline_data = transpose(data(:, 401:500, t));
        T = array2table(baseline_data);
        T.Properties.VariableNames(1:4) = ch_name;
        filepathname = strcat(save_path, "csvdata\\", baseline_csv_name);
        writetable(T, filepathname);

        RT_csv_name=strcat(num2str(s, "%02d"), '_', num2str(t, "%03d"), '_RT.csv');
        RT_data = transpose(data(:, 901:1000, t));
        T = array2table(RT_data);
        T.Properties.VariableNames(1:4) = ch_name;
        filepathname = strcat(save_path, "csvdata\\", RT_csv_name);
        writetable(T, filepathname);
    end
end

%% 8ch
data = ones(8, 1500, 581);

for t=1:581
    data(1, :, t) = EEG.data(5, :, RT_index(t, 3));  % Fz
    data(2, :, t) = EEG.data(4, :, RT_index(t, 3));  % F3
    data(3, :, t) = EEG.data(6, :, RT_index(t, 3));  % F4
    data(4, :, t) = EEG.data(15, :, RT_index(t, 3)); % Cz
    data(5, :, t) = EEG.data(14, :, RT_index(t, 3)); % C3
    data(6, :, t) = EEG.data(16, :, RT_index(t, 3)); % C4
    data(7, :, t) = EEG.data(25, :, RT_index(t, 3)); % Pz
    data(8, :, t) = EEG.data(31, :, RT_index(t, 3)); % Oz
end

data_name = 'Lanekeeping8chforTCDFGCPOWER';
trial=581;
subject=1;
ch_name ={'Fz', 'F3', 'F4', 'Cz', 'C3', 'C4', 'Pz', 'Oz'};
save_path = strcat("I:\\共用雲端硬碟\\CNElab_枋劭勳\\10.交接資料\\Shane-InfoFlowNet\\data\\", data_name, "\\");

for s=1:subject
    for t=1:trial
        baseline_csv_name=strcat(num2str(s, "%02d"), '_', num2str(t, "%03d"), '_baseline_01.csv');
        baseline_data = transpose(data(:, 1:100, t));
        T = array2table(baseline_data);
        T.Properties.VariableNames(1:8) = ch_name;
        filepathname = strcat(save_path, "csvdata\\", baseline_csv_name);
        writetable(T, filepathname);

        baseline_csv_name=strcat(num2str(s, "%02d"), '_', num2str(t, "%03d"), '_baseline_02.csv');
        baseline_data = transpose(data(:, 101:200, t));
        T = array2table(baseline_data);
        T.Properties.VariableNames(1:8) = ch_name;
        filepathname = strcat(save_path, "csvdata\\", baseline_csv_name);
        writetable(T, filepathname);

        baseline_csv_name=strcat(num2str(s, "%02d"), '_', num2str(t, "%03d"), '_baseline_03.csv');
        baseline_data = transpose(data(:, 201:300, t));
        T = array2table(baseline_data);
        T.Properties.VariableNames(1:8) = ch_name;
        filepathname = strcat(save_path, "csvdata\\", baseline_csv_name);
        writetable(T, filepathname);

        baseline_csv_name=strcat(num2str(s, "%02d"), '_', num2str(t, "%03d"), '_baseline_04.csv');
        baseline_data = transpose(data(:, 301:400, t));
        T = array2table(baseline_data);
        T.Properties.VariableNames(1:8) = ch_name;
        filepathname = strcat(save_path, "csvdata\\", baseline_csv_name);
        writetable(T, filepathname);

        baseline_csv_name=strcat(num2str(s, "%02d"), '_', num2str(t, "%03d"), '_baseline_05.csv');
        baseline_data = transpose(data(:, 401:500, t));
        T = array2table(baseline_data);
        T.Properties.VariableNames(1:8) = ch_name;
        filepathname = strcat(save_path, "csvdata\\", baseline_csv_name);
        writetable(T, filepathname);

        RT_csv_name=strcat(num2str(s, "%02d"), '_', num2str(t, "%03d"), '_RT.csv');
        RT_data = transpose(data(:, 901:1000, t));
        T = array2table(RT_data);
        T.Properties.VariableNames(1:8) = ch_name;
        filepathname = strcat(save_path, "csvdata\\", RT_csv_name);
        writetable(T, filepathname);
    end
end

