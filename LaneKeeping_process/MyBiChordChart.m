function MBC = MyBiChordChart(dataMat, ch_name)
    MBC.Sep=1/10;
    MBC.Arrow='off';
    MBC.CData=colormap();
    MBC.ax.NextPlot='add';
    MBC.dataMat=dataMat;
    MBC.ch_name=ch_name;
    
    MBC.ax.XLim=[-1.38,1.38];
    MBC.ax.YLim=[-1.38,1.38];
    MBC.ax.XTick=[];
    MBC.ax.YTick=[];
    MBC.ax.XColor='none';
    MBC.ax.YColor='none';
    MBC.ax.PlotBoxAspectRatio=[1,1,1];
    MBC.LRadius=1.28;
    MBC.thetaSet=[];
    
    numC=size(MBC.dataMat,1);
    ratioC=ones(1, numC) ./ numC;
    ratioC=[0,ratioC];
    
    sepLen=(2*pi*MBC.Sep)./numC;
    baseLen=2*pi*(1-MBC.Sep);
    % 绘制方块
    for i=1:numC
        theta1=sepLen/2+sum(ratioC(1:i))*baseLen+(i-1)*sepLen;
        theta2=sepLen/2+sum(ratioC(1:i+1))*baseLen+(i-1)*sepLen;
        theta=linspace(theta1,theta2,100);
        X=cos(theta);Y=sin(theta);
        MBC.squareHdl(i)=fill([1.05.*X,1.15.*X(end:-1:1)],[1.05.*Y,1.15.*Y(end:-1:1)],...
                    MBC.CData(i,:),'EdgeColor','none');
        theta3=(theta1+theta2)/2;
        MBC.meanThetaSet(i)=theta3;
        MBC.nameHdl(i)=text(cos(theta3).*MBC.LRadius,sin(theta3).*MBC.LRadius,MBC.ch_name{i},'FontSize',14,'FontName','Arial',...
                    'HorizontalAlignment','center','Rotation',-(1.5*pi-theta3)./pi.*180,'Tag','BiChordLabel');

        hold on;
    end

    for i=1:numC
        for j=1:numC
            theta_i_1=sepLen/2+sum(ratioC(1:i))*baseLen+(i-1)*sepLen;
            theta_i_2=sepLen/2+sum(ratioC(1:i+1))*baseLen+(i-1)*sepLen;
            theta_i_3=theta_i_1+(theta_i_2-theta_i_1).*sum(abs(MBC.dataMat(:,i)))./(sum(abs(MBC.dataMat(:,i)))+sum(abs(MBC.dataMat(i,:))));

            theta_j_1=sepLen/2+sum(ratioC(1:j))*baseLen+(j-1)*sepLen;
            theta_j_2=sepLen/2+sum(ratioC(1:j+1))*baseLen+(j-1)*sepLen;
            theta_j_3=theta_j_1+(theta_j_2-theta_j_1).*sum(abs(MBC.dataMat(:,j)))./(sum(abs(MBC.dataMat(:,j)))+sum(abs(MBC.dataMat(j,:))));

            ratio_i_1=MBC.dataMat(i,:);ratio_i_1=[0,ratio_i_1./sum(ratio_i_1)];
            ratio_j_2=MBC.dataMat(:,j)';ratio_j_2=[0,ratio_j_2./sum(ratio_j_2)];
            if true
                theta1=theta_i_2+(theta_i_3-theta_i_2).*sum(ratio_i_1(1:j));
                theta2=theta_i_2+(theta_i_3-theta_i_2).*sum(ratio_i_1(1:j+1));
                theta3=theta_j_3+(theta_j_1-theta_j_3).*sum(ratio_j_2(1:i));
                theta4=theta_j_3+(theta_j_1-theta_j_3).*sum(ratio_j_2(1:i+1));

                tPnt1=[cos(theta1),sin(theta1)];
                tPnt2=[cos(theta2),sin(theta2)];
                tPnt3=[cos(theta3),sin(theta3)];
                tPnt4=[cos(theta4),sin(theta4)];
                MBC.thetaSet=[MBC.thetaSet;theta1;theta2;theta3;theta4];
                MBC.thetaFullSet(i,j)=theta1;
                MBC.thetaFullSet(i,j+1)=theta2;
                MBC.thetaFullSet(j,i+numC)=theta3;
                MBC.thetaFullSet(j,i+numC+1)=theta4;
                
                % 计算贝塞尔曲线
                tLine1=bezierCurve([tPnt1;0,0;tPnt4.*.96],200);
                tLine2=bezierCurve([tPnt2;0,0;tPnt3.*.96],200);
                tline3=[cos(linspace(theta2,theta1,100))',sin(linspace(theta2,theta1,100))'];
                tline4=[cos(theta4).*.96,sin(theta4).*.96;
                    cos(theta3/2+theta4/2).*.99,sin(theta3/2+theta4/2).*.99;
                    cos(theta3).*.96,sin(theta3).*.96];
                v = MBC.dataMat(i, j);
                if v~=0
                    v = round(v*256);
                    MBC.chordMatHdl(i,j)=fill([tLine1(:,1);tline4(:,1);tLine2(end:-1:1,1);tline3(:,1)],...
                        [tLine1(:,2);tline4(:,2);tLine2(end:-1:1,2);tline3(:,2)],...
                        MBC.CData(v,:),'EdgeColor','none');
                end
            else
            end
        end
    end
    function pnts=bezierCurve(pnts,N)
        t=linspace(0,1,N);
        p=size(pnts,1)-1;
        coe1=factorial(p)./factorial(0:p)./factorial(p:-1:0);
        coe2=((t).^((0:p)')).*((1-t).^((p:-1:0)'));
        pnts=(pnts'*(coe1'.*coe2))';
    end
end

