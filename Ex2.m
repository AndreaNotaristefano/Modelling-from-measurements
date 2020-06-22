%% Exercise 2
clear all
clc

%% Compute the training data set
dt=0.01; 
T=10;
t=0:dt:T;
N=64;
tcut=0.6;

input=[]; 
output=[];
jmax=100;

for j=1:jmax  % training
    %x0=randn(N,1);
    x = 0:(2*pi)/(N-1) : 2*pi;
    x0 = -15*randn*sin(x);
    [y,ks,t] = myks(x0',dt,T,tcut);
    input=[input; y(1:end-1,:)];
    output=[output; y(2:end,:)];
   % waterfall(ks,t,y.'), colormap([0 0 0])
   % hold on
   % set(gca,'Xlim',[-50 50])
end

%% Train NN
net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'logsig';
%net.layers{4}.transferFcn = 'tansig';
%net.layers{5}.transferFcn = 'purelin';

%%
net = train(net,input.',output.');

%%
x0 = -15*randn*sin(x);
x0=x0';
[y0,ks0,t0] = myks(x0,dt,T,tcut);

ynn(1,:)=x0;
%ynn(1,:)=-x0(1)*sin(pi*ks0/10);
for jj=2:length(t)
    y0nn=net(x0);
    ynn(jj,:)=y0nn.';  
    x0=y0nn;
end

figure (1)
subplot(2,2,2)
pcolor(ks0,t0,ynn),shading interp, colormap(hot)
title('NN','FontSize',15)
colorbar
xlabel('space','FontSize',15)
ylabel('time','FontSize',15)
caxis([-20 20])


subplot(2,2,1)
pcolor(ks0,t0,y0),shading interp, colormap(hot)
colorbar
title('Computed','FontSize',15)
xlabel('space','FontSize',15)
ylabel('time','FontSize',15)
caxis([-20 20])

%% Low rank NN
r1=3;
r2=10;

[net1] = EX2_NNSVD_sin(r1,T,t,tcut,N,jmax);
[net2] = EX2_NNSVD_sin(r2,T,t,tcut,N,jmax);

[u,s,v] = svd(y0');
y_lr1 = s(1:r1,1:r1)*v(:,1:r1)';
y_lr2 = s(1:r2,1:r2)*v(:,1:r2)';

ynn1=y_lr1(:,1)';
y0nn1 = ynn1';

ynn2=y_lr2(:,1)';
y0nn2 = ynn2';
%ynn(1,:)=-x0(1)*sin(pi*ks0/10);
for jj=2:length(t)
    y0nnl1=net1(y0nn1);
    ynn1(jj,:)=y0nnl1';
    y0nn1 = y0nnl1;
    
    y0nnl2=net2(y0nn2);
    ynn2(jj,:)=y0nnl2';
    y0nn2 = y0nnl2;
end

ynnlr1=u(:,1:r1)*ynn1';
ynnlr2=u(:,1:r2)*ynn2';

figure (1)
subplot(2,2,3)
pcolor(ks0,t0,ynnlr1'),shading interp, colormap(hot)
colorbar
title('r=3 NN','FontSize',15)
xlabel('space','FontSize',15)
ylabel('time','FontSize',15)
caxis([-20 20])

subplot(2,2,4)
pcolor(ks0,t0,ynnlr2'),shading interp, colormap(hot)
colorbar
title('r=10 NN','FontSize',15)
xlabel('space','FontSize',15)
ylabel('time','FontSize',15)
caxis([-20 20])




