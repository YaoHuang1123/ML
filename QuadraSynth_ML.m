%author:  huangyao
%12-4-2023

clc
clear
close all
tic

[data_tr, head] = ;

y = data_tr(:,1);
n = length(y); 

data = data_tr(:,2:end);

[row,col] = size(data);
M = data; 
auc_2 = 0;

s = 6;

n_col = size(M, 2);
C = nchoosek(1:n_col, s);

result = cell(size(C, 1), 1);

for i = 1:size(C, 1)
    result{i} = M(:, C(i, :));
end
%---------------------------------------------------
auc_comb2 = [];
num = size(result,1);

for j = 1:num
    cell_data = result{j};
    
    p_A = cell_data(:,1);
    p_B = cell_data(:,2);
    p_C = cell_data(:,3);
    p_D = cell_data(:,4);
    p_E = cell_data(:,5);
    p_F = cell_data(:,6);
    
    H = [zeros(s), zeros(s,n); zeros(n,s), eye(n)]; 
    f = [zeros(s,1); -ones(n,1)]; 
    Aeq = [1, 1, 1, 1, 1, 1, zeros(1,n)]; 
    beq = 1;
    lb = [0; 0; 0; 0; 0; 0; -inf*ones(n,1)]; 
    ub = [inf; inf; inf; inf; inf; inf; inf*ones(n,1)]; 
    A = [-diag(y)*p_A, -diag(y)*p_B, -diag(y)*p_C, -diag(y)*p_D, -diag(y)*p_E, -diag(y)*p_F, -eye(n)]; %
    b = -y;
    options = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'Display', 'off');
    [x,~,~,~,~] = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],options);
    alpha_A = x(1);
    alpha_B = x(2);
    alpha_C = x(3);
    alpha_D = x(4);
    alpha_E = x(5);
    alpha_F = x(6);
    
    p = alpha_A*p_A + alpha_B*p_B + alpha_C*p_C + alpha_D*p_D + alpha_E*p_E + alpha_F*p_F + x(7:end);
    [X,Y,T,auc_comb] = perfcurve(y,p,1);
    
    auc_comb2(j) = auc_comb;
    
    
end

[~,idex2] = find(auc_comb2==max(auc_comb2));

cell_data = result{idex2};

p_A = cell_data(:,1);
p_B = cell_data(:,2);
p_C = cell_data(:,3);
p_D = cell_data(:,4);
p_E = cell_data(:,5);
p_F = cell_data(:,6);

H = [zeros(s), zeros(s,n); zeros(n,s), eye(n)]; 
f = [zeros(s,1); -ones(n,1)]; 
Aeq = [1, 1, 1, 1, 1, 1, zeros(1,n)]; 
beq = 1; 
lb = [0; 0; 0; 0; 0; 0; -inf*ones(n,1)];
ub = [inf; inf; inf; inf; inf; inf; inf*ones(n,1)]; 
A = [-diag(y)*p_A, -diag(y)*p_B, -diag(y)*p_C, -diag(y)*p_D, -diag(y)*p_E, -diag(y)*p_F, -eye(n)]; %
b = -y;
options = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'Display', 'off');
alpha_A = x(1);
alpha_B = x(2);
alpha_C = x(3);
alpha_D = x(4);
alpha_E = x(5);
alpha_F = x(6);

p = alpha_A*p_A + alpha_B*p_B + alpha_C*p_C + alpha_D*p_D + alpha_E*p_E + alpha_F*p_F + x(7:end);
mdl = fitglm(p, y, 'Distribution', 'binomial');
predictedProbabilities = predict(mdl, p);
p = predictedProbabilities;

L_te = length(pe_B);
y_e = data_te(:,1);
p_e = alpha_A*pe_A+alpha_B*pe_B + alpha_C*pe_C + alpha_D*pe_D + alpha_E*pe_E + alpha_F*pe_F + x(7:L_te+7-1);
md2 = fitglm(p_e, y_e, 'Distribution', 'binomial');
predictedProbabilities2 = predict(md2, p_e);
p_e = predictedProbabilities2;

[X,Y,T,auc_comb] = perfcurve(y,p,1);

[X1, Y1, T1, auc1] = perfcurve(y, p_A, 1);
[X2, Y2, T2, auc2] = perfcurve(y, p_B, 1);
[X3, Y3, T3, auc3] = perfcurve(y, p_C, 1);
[X4, Y4, T4, auc4] = perfcurve(y, p_D, 1);
[X5, Y5, T5, auc5] = perfcurve(y, p_E, 1);
[X6, Y6, T6, auc6] = perfcurve(y, p_F, 1);

fprintf('AUC of Model 1: %.4f\n', auc1);
fprintf('AUC of Model 2: %.4f\n', auc2);
fprintf('AUC of Model 3: %.4f\n', auc3);
fprintf('AUC of Model 4: %.4f\n', auc4);
fprintf('AUC of Model 5: %.4f\n', auc5);
fprintf('AUC of Model 6: %.4f\n', auc6);
fprintf('AUC of Combined Model: %.4f\n', auc_comb);

toc
