clc; clear; %close all;
load('../datasets/train_hands.mat');
% load('23.train.mat');


% For CMU-mocap subjects
% S_new = [];
% W_new = [];
%  for j = 1:length(S)/3
% S_new = vertcat(S_new, transpose(S(3*j - 2 : 3*j,:)));
% W_new = vertcat(W_new, transpose(W(2*j - 1 : 2*j,:)));
%  end
% S = S_new; W = W_new; clear W_new S_new;

keypoints = 21;

final_S = [];
final_W = [];
diff = [];
further_factor = 0.4;

for idx = 1: (length(S)/keypoints)
    
    temp_S = S(keypoints*idx - (keypoints - 1) : keypoints*idx, :);
    temp_S = mean_centering(temp_S);       
    final_S = vertcat(final_S, temp_S);
    
    temp_W = W(keypoints*idx - (keypoints-1) : keypoints*idx, :);
    temp_W = mean_centering(temp_W);    
    final_W = vertcat(final_W, temp_W);    
    
    disp(idx);
    
end

% Mean centering and normalizing cuts a part of my image but at least
% reduces my 2D error by a lot

% The best is just mean centering in both

% w1 = s1(:, 1:2);
save('W_hand_train.mat', 'final_W')
save('S_hand_train.mat', 'final_S')

plot_with_specs(w1)


function plot_with_specs(matrix)
    size_matrix = size(matrix);
    if size_matrix(2) == 2
        plot(matrix(:, 1), matrix(:, 2), 'r.', 'MarkerSize', 25);
    else
        plot3(matrix(:, 1), matrix(:, 2), matrix(:, 3), 'b.', 'MarkerSize', 25);
    end
    set(gca, 'FontSize', 25); hold on;    
end

function out_matrix = mean_centering(matrix)
    size_matrix = size(matrix);
    out_matrix(:, 1) = matrix(:, 1) - mean(matrix(:, 1));
    out_matrix(:, 2) = matrix(:, 2) - mean(matrix(:, 2));
    if size_matrix(2) == 3
        out_matrix(:, 3) = matrix(:, 3) - mean(matrix(:, 3));
    end
end

function w1 = normalize_mat(w1)
    size_matrix = size(w1);
    w1(:,1) = w1(:,1)/norm(w1(:,1), 1);
    w1(:,2) = w1(:,2)/norm(w1(:,2), 1);
%     w1(:,1) = w1(:,1)/norm(w1(:,1));
%     w1(:,2) = w1(:,2)/norm(w1(:,2));    
    if size_matrix(2) == 3
        w1(:,3) = w1(:,3)/norm(w1(:,3));
    end
end

function w1 = further_process(w1, factor)
    w1(:, 1) = w1(:, 1).*factor;
    w1(:, 2) = w1(:, 2).*factor;
end
