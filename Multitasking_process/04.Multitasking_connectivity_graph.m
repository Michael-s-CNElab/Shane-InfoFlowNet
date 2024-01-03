RT_window_feature = mean(mean(mean(corr2_corr1, 6), 5), 4);
RT_window_feature = RT_window_feature(:, :, 7);

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