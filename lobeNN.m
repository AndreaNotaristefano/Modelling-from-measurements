function [net] = lobeNN(jmax,x0_vect,t,step_before,Lorenz)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
input=[]; output=[];
m=-1;
for j=1:jmax  % training trajectories
    x0 = x0_vect(:,j);
    [t,y] = ode45(Lorenz,t,x0);
    rightLine = (y(:,2)>m*y(:,1));
    leftLine = ~rightLine;

    clas_vector=zeros(length(rightLine),1);
    clas_vector(rightLine==1) = 1;
    clas_vector(rightLine==0) = -1;
    jump = abs(clas_vector(1:end-1)-clas_vector(2:end))/2;
    
    input=[input; [y(1:end-step_before,:)]];
    output=[output; jump(step_before:end)];
end

net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';

net = train(net,input.',output.');
end

