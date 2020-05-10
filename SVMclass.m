%% author Zavieton MATLAB2018b
% 2020/4
clear;
clc;

%%
% Set up fittype and options.
ft = fittype( 'fourier1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Normalize = 'on';
opts.StartPoint = [0 0 0 7.26160255574314];


%% 载入数据
fprintf('-----正在载入数据-----\n\n');
data0 = importdata('train_class0.mat');
data1 = importdata('train_class1.mat');
data2 = importdata('train_class2.mat');
data3 = importdata('train_class3.mat');
data4 = importdata('train_class4.mat');
data5 = importdata('train_class5.mat');
data6 = importdata('train_class6.mat');
data7 = importdata('train_class7.mat');
data8 = importdata('train_class8.mat');
data9 = importdata('train_class9.mat');
data10 = importdata('train_class10.mat');
data11 = importdata('train_class11.mat');
data12 = importdata('train_class12.mat');
data13 = importdata('train_class13.mat');
data14 = importdata('train_class14.mat');
data15 = importdata('train_class15.mat');
data16 = importdata('train_class16.mat');
data17 = importdata('train_class17.mat');
    hist(data0(1,:),-5:0.1:5);
    figure;
    hist(data1(1,:),-5:0.1:5);

fprintf('-----正在提取特征-----\n\n');
%% 特征提取

G = 120; %取中值滤波阶数为G
f = 1;
t = 1:f:2048*f;
% 中值滤波
md0 = medfilt1(data0,G);
md1 = medfilt1(data1,G);
md2 = medfilt1(data2,G);
md3 = medfilt1(data3,G);
md4 = medfilt1(data4,G);
md5 = medfilt1(data5,G);
md6 = medfilt1(data6,G);
md7 = medfilt1(data7,G);
md8 = medfilt1(data8,G);
md9 = medfilt1(data9,G);
md10 = medfilt1(data10,G);
md11 = medfilt1(data11,G);
md12 = medfilt1(data12,G);
md13 = medfilt1(data13,G);
md14 = medfilt1(data14,G);
md15 = medfilt1(data15,G);
md16 = medfilt1(data16,G);
md17 = medfilt1(data17,G);
C_end = 2048;
t_c = 1*f:f:C_end*f;

%取样中间部分样本进行处理
mc0 = md0(1:C_end,:);mc10 = md10(1:C_end,:);
mc2 = md2(1:C_end,:);mc11 = md11(1:C_end,:);
mc3 = md3(1:C_end,:);mc12 = md12(1:C_end,:);
mc4 = md4(1:C_end,:);mc13 = md13(1:C_end,:);
mc5 = md5(1:C_end,:);mc14 = md14(1:C_end,:);
mc6 = md6(1:C_end,:);mc15 = md15(1:C_end,:);
mc7 = md7(1:C_end,:);mc16 = md16(1:C_end,:);
mc8 = md8(1:C_end,:);mc17 = md17(1:C_end,:);
mc9 = md9(1:C_end,:);mc1 = md1(1:C_end,:);
for i=1:500
    fprintf("当前进度 %d/500 \n",i);
    % Fit model to data.
    % 采用傅里叶变换拟合数据
    [fitresult0, ~] = fit( t_c', mc0(:,i), ft, opts );
    A0(i) = fitresult0.a1;
    B0(i) = fitresult0.b1;
    W0(i) = fitresult0.w;
    S0(i) = fitresult0.a0;
    
    [fitresult1, ~] = fit( t_c', mc1(:,i), ft, opts );
    A1(i) = fitresult1.a1;
    B1(i) = fitresult1.b1;
    W1(i) = fitresult1.w;
    S1(i) = fitresult1.a0;
    
    [fitresult2, ~] = fit( t_c', mc2(:,i), ft, opts );
    A2(i) = fitresult2.a1;
    B2(i) = fitresult2.b1;
    W2(i) = fitresult2.w;
    S2(i) = fitresult2.a0;
    
    [fitresult3, ~] = fit( t_c', mc3(:,i), ft, opts );
    A3(i) = fitresult3.a1;
    B3(i) = fitresult3.b1;
    W3(i) = fitresult3.w;
    S3(i) = fitresult3.a0;
    
    [fitresult4, ~] = fit( t_c', mc4(:,i), ft, opts );
    A4(i) = fitresult4.a1;
    B4(i) = fitresult4.b1;
    W4(i) = fitresult4.w;
    S4(i) = fitresult4.a0;
    
    [fitresult5, ~] = fit( t_c', mc5(:,i), ft, opts );
    A5(i) = fitresult5.a1;
    B5(i) = fitresult5.b1;
    W5(i) = fitresult5.w;
    S5(i) = fitresult5.a0;
    
    [fitresult6, ~] = fit( t_c', mc6(:,i), ft, opts );
    A6(i) = fitresult6.a1;
    B6(i) = fitresult6.b1;
    W6(i) = fitresult6.w;
    S6(i) = fitresult6.a0;

    [fitresult7, ~] = fit( t_c', mc7(:,i), ft, opts );
    A7(i) = fitresult7.a1;
    B7(i) = fitresult7.b1;
    W7(i) = fitresult7.w;
    S7(i) = fitresult7.a0;
    
    [fitresult8, ~] = fit( t_c', mc8(:,i), ft, opts );
    A8(i) = fitresult8.a1;
    B8(i) = fitresult8.b1;
    W8(i) = fitresult8.w;
    S8(i) = fitresult8.a0;
    
    [fitresult9, ~] = fit( t_c', mc9(:,i), ft, opts );
    A9(i) = fitresult9.a1;
    B9(i) = fitresult9.b1;
    W9(i) = fitresult9.w;
    S9(i) = fitresult9.a0;
    
    [fitresult10, ~] = fit( t_c', mc10(:,i), ft, opts );
    A10(i) = fitresult10.a1;
    B10(i) = fitresult10.b1;
    W10(i) = fitresult10.w;
    S10(i) = fitresult10.a0;
    
    [fitresult11, ~] = fit( t_c', mc11(:,i), ft, opts );
    A11(i) = fitresult11.a1;
    B11(i) = fitresult11.b1;
    W11(i) = fitresult11.w;
    S11(i) = fitresult11.a0;
    
    [fitresult12, ~] = fit( t_c', mc12(:,i), ft, opts );
    A12(i) = fitresult12.a1;
    B12(i) = fitresult12.b1;
    W12(i) = fitresult12.w;
    S12(i) = fitresult12.a0;
    
    [fitresult13, ~] = fit( t_c', mc13(:,i), ft, opts );
    A13(i) = fitresult13.a1;
    B13(i) = fitresult13.b1;
    W13(i) = fitresult13.w;
    S13(i) = fitresult13.a0;
    
    [fitresult14, ~] = fit( t_c', mc14(:,i), ft, opts );
    A14(i) = fitresult14.a1;
    B14(i) = fitresult14.b1;
    W14(i) = fitresult14.w;
    S14(i) = fitresult14.a0;
    
    [fitresult15, ~] = fit( t_c', mc15(:,i), ft, opts );
    A15(i) = fitresult15.a1;
    B15(i) = fitresult15.b1;
    W15(i) = fitresult15.w;
    S15(i) = fitresult15.a0;
    
    [fitresult16, ~] = fit( t_c', mc16(:,i), ft, opts );
    A16(i) = fitresult16.a1;
    B16(i) = fitresult16.b1;
    W16(i) = fitresult16.w;
    S16(i) = fitresult16.a0;
    
    [fitresult17, ~] = fit( t_c', mc17(:,i), ft, opts );
    A17(i) = fitresult17.a1;
    B17(i) = fitresult17.b1;
    W17(i) = fitresult17.w;
    S17(i) = fitresult17.a0;
    

    
    A0(i) = (A0(i)^2 + B0(i)^2)^0.5;
    A1(i) = (A1(i)^2 + B1(i)^2)^0.5;
    A2(i) = (A2(i)^2 + B2(i)^2)^0.5;
    A3(i) = (A3(i)^2 + B3(i)^2)^0.5;
    A4(i) = (A4(i)^2 + B4(i)^2)^0.5;
    A5(i) = (A5(i)^2 + B5(i)^2)^0.5;
    A6(i) = (A6(i)^2 + B6(i)^2)^0.5;
    A7(i) = (A7(i)^2 + B7(i)^2)^0.5;
    A8(i) = (A8(i)^2 + B8(i)^2)^0.5;
    A9(i) = (A9(i)^2 + B9(i)^2)^0.5;
    A10(i) = (A10(i)^2 + B10(i)^2)^0.5;
    A11(i) = (A11(i)^2 + B11(i)^2)^0.5;
    A12(i) = (A12(i)^2 + B12(i)^2)^0.5;
    A13(i) = (A13(i)^2 + B13(i)^2)^0.5;
    A14(i) = (A14(i)^2 + B14(i)^2)^0.5;
    A15(i) = (A15(i)^2 + B15(i)^2)^0.5;
    A16(i) = (A16(i)^2 + B16(i)^2)^0.5;
    A17(i) = (A17(i)^2 + B17(i)^2)^0.5;    
    
end

for j = 1:500
    x0 = data0(:,j); 
    x1 = data1(:,j); 
    x2 = data2(:,j); 
    x3 = data3(:,j); 
    x4 = data4(:,j); 
    x5 = data5(:,j); 
    x6 = data6(:,j); 
    x7 = data7(:,j); 
    x8 = data8(:,j); 
    x9 = data9(:,j); 
    x10 = data10(:,j);
    x11 = data11(:,j); 
    x12 = data12(:,j); 
    x13 = data13(:,j); 
    x14 = data14(:,j); 
    x15 = data15(:,j); 
    x16 = data16(:,j);
    x17 = data17(:,j); 

%求基本频率及幅值
    [Fn0(j),An0(j)] = DFT(x0);
    [Fn1(j),An1(j)] = DFT(x1);
    [Fn2(j),An2(j)] = DFT(x2);
    [Fn3(j),An3(j)] = DFT(x3);
    [Fn4(j),An4(j)] = DFT(x4);
    [Fn5(j),An5(j)] = DFT(x5);
    [Fn6(j),An6(j)] = DFT(x6);
    [Fn7(j),An7(j)] = DFT(x7);
    [Fn8(j),An8(j)] = DFT(x8);
    [Fn9(j),An9(j)] = DFT(x9);
    [Fn10(j),An10(j)] = DFT(x10);
    [Fn11(j),An11(j)] = DFT(x11);
    [Fn12(j),An12(j)] = DFT(x12);
    [Fn13(j),An13(j)] = DFT(x13);
    [Fn14(j),An14(j)] = DFT(x14);
    [Fn15(j),An15(j)] = DFT(x15);
    [Fn16(j),An16(j)] = DFT(x16);
    [Fn17(j),An17(j)] = DFT(x17);
%求均值
    En0(j) = mean(x0);    En1(j) = mean(x1);    En2(j) = mean(x2);    En3(j) = mean(x3);
    En4(j) = mean(x4);    En5(j) = mean(x5);    En6(j) = mean(x6);    En7(j) = mean(x7);
    En8(j) = mean(x8);    En9(j) = mean(x9);    En11(j) = mean(x11);    En12(j) = mean(x12);
    En13(j) = mean(x13);    En14(j) = mean(x14);    En15(j) = mean(x15);    En16(j) = mean(x16);
    En17(j) = mean(x17);   En10(j) = mean(x10);
%求方差
    Dn0(j) = var(x0);    Dn5(j) = var(x5);    Dn10(j) = var(x10);    Dn14(j) = var(x14);
    Dn1(j) = var(x1);    Dn6(j) = var(x6);    Dn11(j) = var(x11);    Dn15(j) = var(x15);
    Dn2(j) = var(x2);    Dn7(j) = var(x7);    Dn12(j) = var(x12);    Dn16(j) = var(x16);
    Dn3(j) = var(x3);    Dn8(j) = var(x8);    Dn13(j) = var(x13);    Dn17(j) = var(x17);
    Dn4(j) = var(x4);    Dn9(j) = var(x9);

%估计振荡次数           
    temp1(j) = 0;   temp2(j) = 0;    temp3(j) = 0; 
    temp4(j) = 0;   temp5(j) = 0;    temp6(j) = 0;
    temp7(j) = 0;    temp8(j) = 0;    temp9(j) = 0;
    temp10(j) = 0;    temp11(j) = 0;    temp12(j) = 0;
    temp13(j) = 0;    temp14(j) = 0;    temp15(j) = 0;    
    temp16(j) = 0;     temp17(j) = 0;    temp18(j) = 0;
    for k=1:2047
        if((x0(k)-En0(j))*(x0(k+1)-En0(j))<0)
            temp1(j) = temp1(j) + 1;
        end
        if((x1(k)-En1(j))*(x1(k+1)-En1(j))<0)
            temp2(j) = temp2(j) + 1;
        end
        if((x2(k)-En2(j))*(x2(k+1)-En2(j))<0)
            temp3(j) = temp3(j) + 1;
        end
        if((x3(k)-En3(j))*(x3(k+1)-En3(j))<0)
            temp4(j) = temp4(j) + 1;
        end
        if((x4(k)-En4(j))*(x4(k+1)-En4(j))<0)
            temp5(j) = temp5(j) + 1;
        end
        if((x5(k)-En5(j))*(x5(k+1)-En5(j))<0)
            temp6(j) = temp6(j) + 1;
        end
        if((x6(k)-En6(j))*(x6(k+1)-En6(j))<0)
            temp7(j) = temp7(j) + 1;
        end
        if((x7(k)-En7(j))*(x7(k+1)-En7(j))<0)
            temp8(j) = temp8(j) + 1;
        end
        if((x8(k)-En8(j))*(x8(k+1)-En8(j))<0)
            temp9(j) = temp9(j) + 1;
        end
        if((x9(k)-En9(j))*(x9(k+1)-En9(j))<0)
            temp10(j) = temp10(j) + 1;
        end
        if((x10(k)-En10(j))*(x10(k+1)-En10(j))<0)
            temp11(j) = temp11(j) + 1;
        end
        if((x11(k)-En11(j))*(x11(k+1)-En11(j))<0)
            temp12(j) = temp12(j) + 1;
        end
        if((x12(k)-En12(j))*(x12(k+1)-En12(j))<0)
            temp13(j) = temp13(j) + 1;
        end
        if((x13(k)-En13(j))*(x13(k+1)-En13(j))<0)
            temp14(j) = temp14(j) + 1;
        end
        if((x14(k)-En14(j))*(x14(k+1)-En14(j))<0)
            temp15(j) = temp15(j) + 1;
        end
        if((x15(k)-En15(j))*(x15(k+1)-En15(j))<0)
            temp16(j) = temp16(j) + 1;
        end
        if((x16(k)-En16(j))*(x16(k+1)-En16(j))<0)
            temp17(j) = temp17(j) + 1;
        end
        if((x17(k)-En17(j))*(x17(k+1)-En17(j))<0)
            temp18(j) = temp18(j) + 1;
        end
    end
end

    D0 = [Fn0;An0;temp1;En0;Dn0;A0;W0;S0]';
    D1 = [Fn1;An1;temp2;En1;Dn1;A1;W1;S1]';
    D2 = [Fn2;An2;temp3;En2;Dn2;A2;W2;S2]';
    D3 = [Fn3;An3;temp4;En3;Dn3;A3;W3;S3]';
    D4 = [Fn4;An4;temp5;En4;Dn4;A4;W4;S4]';
    D5 = [Fn5;An5;temp6;En5;Dn5;A5;W5;S5]';
    D6 = [Fn6;An6;temp7;En6;Dn6;A6;W6;S6]';
    D7 = [Fn7;An7;temp8;En7;Dn7;A7;W7;S7]';
    D8 = [Fn8;An8;temp9;En8;Dn8;A8;W8;S8]';
    D9 = [Fn9;An9;temp10;En9;Dn9;A9;W9;S9]';
    D10 = [Fn10;An10;temp11;En10;Dn10;A10;W10;S10]';
    D11 = [Fn11;An11;temp12;En11;Dn11;A11;W11;S11]';
    D12 = [Fn12;An12;temp13;En12;Dn12;A12;W12;S12]';
    D13 = [Fn13;An13;temp14;En13;Dn13;A13;W13;S13]';
    D14 = [Fn14;An14;temp15;En14;Dn14;A14;W14;S14]';
    D15 = [Fn15;An15;temp16;En15;Dn15;A15;W15;S15]';
    D16 = [Fn16;An16;temp17;En16;Dn16;A16;W16;S16]';
    D17 = [Fn17;An17;temp18;En17;Dn17;A17;W17;S17]';
    
    
 %% 训练数据
fprintf('-----开始整合训练数据并制定标签-----\n\n');
train_num = 450;
test_num = 500 - train_num;
train_data = [D0(1:train_num,:);D1(1:train_num,:);D2(1:train_num,:);D3(1:train_num,:);D4(1:train_num,:);D5(1:train_num,:);
    D6(1:train_num,:);D7(1:train_num,:);D8(1:train_num,:);D9(1:train_num,:);D10(1:train_num,:);D11(1:train_num,:);
    D12(1:train_num,:);D13(1:train_num,:);D14(1:train_num,:);D15(1:train_num,:);D16(1:train_num,:);D17(1:train_num,:)];
exam_data = [D0(train_num+1:500,:);D1(train_num+1:500,:);D2(train_num+1:500,:);D3(train_num+1:500,:);D4(train_num+1:500,:);D5(train_num+1:500,:);
    D6(train_num+1:500,:);D7(train_num+1:500,:);D8(train_num+1:500,:);D9(train_num+1:500,:);D10(train_num+1:500,:);D11(train_num+1:500,:);
    D12(train_num+1:500,:);D13(train_num+1:500,:);D14(train_num+1:500,:);D15(train_num+1:500,:);D16(train_num+1:500,:);D17(train_num+1:500,:)];
% 制定标签
group_train = [0*ones(train_num,1);1*ones(train_num,1);2*ones(train_num,1);3*ones(train_num,1);4*ones(train_num,1);5*ones(train_num,1);6*ones(train_num,1);
    7*ones(train_num,1);8*ones(train_num,1);9*ones(train_num,1);10*ones(train_num,1);11*ones(train_num,1);12*ones(train_num,1);13*ones(train_num,1);14*ones(train_num,1);
    15*ones(train_num,1);16*ones(train_num,1);17*ones(train_num,1)];
exam_group_train = [0*ones(test_num,1);1*ones(test_num,1);2*ones(test_num,1);3*ones(test_num,1);4*ones(test_num,1);5*ones(test_num,1);6*ones(test_num,1);
    7*ones(test_num,1);8*ones(test_num,1);9*ones(test_num,1);10*ones(test_num,1);11*ones(test_num,1);12*ones(test_num,1);13*ones(test_num,1);14*ones(test_num,1);
    15*ones(test_num,1);16*ones(test_num,1);17*ones(test_num,1)];
fprintf('-----开始载入测试数据-----\n\n');
test_data = importdata('test_data.mat');
fprintf('-----开始处理测试数据-----\n\n');
md = medfilt1(test_data,G);
mc = md;
for i=1:34200
    fprintf("当前进度 %d/34200 \n",i);
    [fitresult, ~] = fit( t_c', mc(:,i), ft, opts );
    A_t(i) = fitresult.a1;
    B_t(i) = fitresult.b1;
    W_t(i) = fitresult.w;
    D_t(i) = fitresult.a0;
    
    AA_t(i) = (A_t(i)^2 + B_t(i)^2)^0.5;
    x_t = test_data(:,i); 
    En_t(i) = mean(x_t);
    f(i) = 0; 
    [Fn_t(i),An_t(i)] = DFT(x_t);
    Dn_t(i) = var(x_t);
    for k=1:2047
        if((x_t(k)-Dn_t(i))*(x_t(k+1)-Dn_t(i))<0)
            f(i) = f(i) + 1;
        end
    end
end
test_features = [Fn_t;An_t;f;En_t;Dn_t;AA_t;W_t;D_t]';

% 数据归一化
[Train_matrix,PS] = mapminmax(train_data');
Train_matrix = Train_matrix';
Test_matrix = mapminmax('apply',exam_data',PS);
Test_matrix = Test_matrix';
Task_matrix = mapminmax('apply',test_features',PS);
Task_matrix = Task_matrix';

%% SVM创建/训练(RBF核函数)
% 寻找最佳c/g参数——交叉验证方法
fprintf('-----开始寻找最优参数-----\n\n');
[c,g] = meshgrid(-10:1:10,-10:1:10);
[m,n] = size(c);
cg = zeros(m,n);
eps = 10^(-4);
v = 5;
bestc = 1;
bestg = 0.1;
bestacc = 0;
for i = 1:m
    for j = 1:n
        fprintf("i = %d,j = %d\n",i,j);
        cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j))];
        cg(i,j) = svmtrain(group_train,Train_matrix,cmd);      %#ok<*SVMTRAIN>
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end        
        if abs( cg(i,j)-bestacc )<=eps && bestc > 2^c(i,j) 
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end               
    end
end
% 迭代参考值为bestc=181.0193；bestg=8
%% 创建/训练SVM模型
% bestc = 181.0193;
% bestg = 8;
bestc = 64;
bestg = 2;
cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg)];
fprintf('-----开始创建/训练SVM模型-----\n\n');
model = svmtrain(group_train,Train_matrix,cmd);
% SVM仿真测试
[predict_label_1,accuracy_1] = svmpredict(group_train,Train_matrix,model);
[predict_label_2,accuracy_2] = svmpredict(exam_group_train,Test_matrix,model);
result_1 = [group_train predict_label_1];
result_2 = [exam_group_train predict_label_2];
% 结果预测
[predict_label,accuracy] = svmpredict(ones(34200,1),Task_matrix,model);

%% 做出基于特征的散点图
fprintf('-----开始作图-----\n\n');

figure(1);
subplot(121);
gscatter(Train_matrix(:,3),Train_matrix(:,4),group_train);
title('训练数据样本分布');
xlabel('N（振荡次数）');
ylabel('EX');
grid on;
subplot(122);
gscatter(Train_matrix(:,3),Train_matrix(:,4),predict_label_1);
title('测试数据样本预测分布');
xlabel('N（振荡次数）');
ylabel('EX');
grid on;
figure(2);
subplot(121);
gscatter(Test_matrix(:,3),Test_matrix(:,4),exam_group_train);
title('训练数据样本分布');
xlabel('N（振荡次数）');
ylabel('EX');
grid on;
subplot(122);
gscatter(Test_matrix(:,3),Test_matrix(:,4),predict_label_2);
title('测试数据样本预测分布');
xlabel('N（振荡次数）');
ylabel('EX');
grid on;
figure(3);
gscatter(Task_matrix(:,3),Task_matrix(:,4),predict_label);
title('任务样本分布');
xlabel('N（振荡次数）');
ylabel('EX');
grid on;
%%
function [An,Fn] = DFT(x)
fs=2000000; % 采样频率，自己根据实际情况设置
N=length(x); % x 是待分析的数据
n=1:N;
%1-FFT
X=fft(x); % FFT
X=X(1:N/2);
Xabs=abs(X);
Xabs(1) = 0; %直流分量置0
for i= 1 : 2 
    [~,index]=max(Xabs);
    if(Xabs(index-1) > Xabs(index+1))
        a1 = Xabs(index-1) / Xabs(index);
        r1 = 1/(1+a1);
        k01 = index -1;
    else
        a1 = Xabs(index) / Xabs(index+1);
        r1 = 1/(1+a1);
        k01 = index;
    end
end
Fn = (k01+r1-1)*fs/N; %基波频率
An= 2*pi*r1*Xabs(k01)/(N*sin(r1*pi)); %基波幅值
end