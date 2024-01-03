%% generate data
channel=3;
sampling_rate=500;
trial=100;

data = ones(channel, sampling_rate, trial);

f1=35;
fs=1000;
N=500;
n=0:N-1;
t=n/fs;
y=4*sin(2*pi*f1*t);

data(3, :, :) = normrnd(0, 1, [1 sampling_rate trial]);

for a=1:trial
    data(1, :, a) = y;
    data(2, :, a) = 4*sawtooth(2*pi*50*t);
end

for i=1:trial
    for t=16:sampling_rate
        data(1, t, i) = data(1, t, i) + data(2, t-10, i) * normpdfg(t,100,8,250,0.5);
        data(3, t, i) = data(3, t, i) + data(2, t-10, i) * normpdfg(t,100,8,250,0.5);
    end
end

% EEG.data = data;
% eeglab redraw;

%% generate label and save to csvfile

data_name = 's13';
trial=100;
window_size=100;
overlap=1;
subject=1;
samplingrate = 500;
ch_name = {'sine' 'sawtooth' 'random'};
save_path = strcat("I:\\共用雲端硬碟\\CNElab_枋劭勳\\10.交接資料\\Shane-InfoFlowNet\\data\\", data_name, "\\");

max_count = ((length(data)-window_size)/overlap)+1;

label={};

WOI = 151:1:350;

for s=1:subject
    for t=1:trial
        csv_name=strcat(num2str(s, "%02d"), '_', num2str(t, "%03d"), '.csv');
        trial_data = transpose(data(:, :, t));
        T = array2table(trial_data);
        T.Properties.VariableNames(1:3) = ch_name;
        filepathname = strcat(save_path, "csvdata\\", csv_name);
        writetable(T, filepathname);
        for c=1:max_count

            label{end+1, 1}=s;
            label{end, 2}=t;

            temp = data(:, c:c+window_size-1, t);
    
            win_start_pt = ((c-1)*overlap)+1;
            win_end_pt = win_start_pt+window_size-1;

            label{end, 3}=win_start_pt;
            label{end, 4}=win_end_pt;

            label{end, 5}=win_start_pt / samplingrate;
            label{end, 6}=win_end_pt / samplingrate;

            if find(win_start_pt==WOI) & find(win_end_pt==WOI)
                label{end, 7}=1;
            else
                label{end, 7}=0;
            end
            label{end, 8}=csv_name;
        end
    end
end

T = array2table(label);
T.Properties.VariableNames(1:8) = {'subject' 'trial'...
                    'window start pt' 'window end pt'...
                    'window start time' 'window end time' 'WOI', 'csvname'};
writetable(T, strcat(save_path, data_name, "_overlap1_label.csv"));