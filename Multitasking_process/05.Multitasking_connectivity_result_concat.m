windows = 15;
subject = 30;
channel = 4;
trial = 36;
shuffle = 30;
ch_location = {'Fz', 'Cz', 'Pz', 'Oz'};

result_path = dir(['Multitasking_head8_eye_4ch_result\', '*.mat']);

for i=1:length(result_path)
    filename_sp = strsplit(result_path(i).name, ["_" "."]);
    sub = str2num(filename_sp{1});
    tr = str2num(filename_sp{2});
    w = floor(str2num(filename_sp{3})/100)+1;
end

corr2_corr1 = ones(channel, channel, windows, trial, subject, shuffle);
xcorr2_xcorr1 = ones(channel, channel, windows, trial, subject, shuffle);
cos2_cos1 = ones(channel, channel, windows, trial, subject, shuffle);