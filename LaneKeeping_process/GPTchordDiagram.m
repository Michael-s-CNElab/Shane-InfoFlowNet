function chordDiagram(matrix)
    numNodes = size(matrix, 1);
    theta = linspace(0, 2 * pi, numNodes + 1);
    [x, y] = pol2cart(theta, 1);
    
    figure;
    hold on;
    axis equal;
    axis off;
    
    for i = 1:numNodes
        for j = i:numNodes
            if matrix(i, j) > 0
                drawChord([x(i), y(i)], [x(j), y(j)], matrix(i, j));
            end
        end
    end
    
    for i = 1:numNodes
        plot([x(i), x(i+1)], [y(i), y(i+1)], 'k', 'LineWidth', 2);
    end
    
    hold off;
end

function drawChord(p1, p2, weight)
    t = linspace(0, 1, 100);
    b = (1-t).^3;
    c = 3*t.*(1-t).^2;
    d = 3*t.^2.*(1-t);
    e = t.^3;
    B = [b; c; d; e];
    controlPoints = [p1; (p1 + p2)/2; (p1 + p2)/2; p2];
    curve = B' * controlPoints;
    plot(curve(:, 1), curve(:, 2), 'b', 'LineWidth', weight);
end