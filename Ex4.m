clear all
clc

load('BZ.mat')
[m,n,k]=size(BZ_tensor); % x vs y vs time data
% for j=1:k
%     A=BZ_tensor(:,:,j);
%     pcolor(A), shading interp, pause(0.2)
% end

%% SVD
r = 160; %rank

X = reshape(BZ_tensor,[prod(size(BZ_tensor(:,:,1))), length(BZ_tensor(1,1,:))]);
[U,S,V] = svd(X,'econ');
semilogy(diag(S)./sum(diag(S)),'o-','linewidth',2)

set(gca, 'GridLineStyle','--','FontSize',35)
ylabel('Singular value \sigma_r weight','FontSize',40,'FontWeight','bold')
xlabel('r','FontSize',40,'FontWeight','bold')
hold on
semilogy(r,S(r,r)./sum(diag(S)),'ro','linewidth',8)
grid on
text(r,S(r,r)./sum(diag(S))+2*10^-4,'Truncation r=160','color','red','FontSize',40,'FontWeight','bold')

%% Reconstrution from a low rank 

Ut=U(:,1:r); 
St=S(1:r,1:r); 
Vt=V(:,1:r);

X_approx = Ut*St*Vt';
X_approx= X_approx';

for i=1:length(X(1,:))
    X_vec_approx(:,:,i)=reshape(X_approx(i,:),size(BZ_tensor(:,:,1)));
end
figure (1)
subplot(1,2,2)
pcolor(X_vec_approx(:,:,1)); shading interp; colormap(hot)
title('SVD approximation','FontSize',40)
colorbar
set(gca, 'FontSize',35)
%set(gcf, 'Renderer', 'painters')
subplot(1,2,1)
set(gca, 'FontSize',35)
pcolor(BZ_tensor(:,:,1)); shading interp; colormap(hot)
title('Truth','FontSize',40)
colorbar
set(gca, 'FontSize',35)
%set(gcf, 'Renderer', 'painters')


fileIDa=fopen('Homework/EX4svd.dat','w'); % apre file scrittura risultati
fprintf(fileIDa,'VARIABLES=x,y,us,uaprox\n');
fprintf(fileIDa,strcat('ZONE T="',fileIDa,'"\n'));
cont=[length(BZ_tensor(:,1,1));length(BZ_tensor(1,:,1))];
fprintf(fileIDa,'I=%u J=%u F=POINT\n',cont);
write=[reshape([1:length(BZ_tensor(:,1,1))]'.*ones(size(BZ_tensor(:,:,1))),[prod(size(BZ_tensor(:,:,1))) 1]),reshape([1:length(BZ_tensor(1,:,1))].*ones(size(BZ_tensor(:,:,1))),[prod(size(BZ_tensor(:,:,1))) 1]),reshape(BZ_tensor(:,:,1),[prod(size(BZ_tensor(:,:,1))) 1]),reshape(X_vec_approx(:,:,1),[prod(size(X_vec_approx(:,:,1))) 1])];
fprintf(fileIDa,'%8.6f %8.6f %8.6f %8.6f\r\n',write');
clear write
fclose(fileIDa);
clear fileIDa 

Xtilde = St*Vt';

%% NN on the low rank subspace
jmax = 1100; % time of training

input=[]; 
output=[];

for j=1:jmax  % training
    input=[input; Xtilde(:,j)];
    output=[output; Xtilde(:,j+1)];
end

%%
net1 = feedforwardnet([20 20 20]);
net1.layers{1}.transferFcn = 'tansig';
net1.layers{2}.transferFcn = 'logsig';
net1.layers{3}.transferFcn = 'radbas';


%%
net1 = train(net1,input.',output.');

%% NN SVD prediction
jpred = 1100;
XtildeNN = net1(Xtilde(:,jpred+1)');

%%
XNN=[];
y0=Xtilde(:,1)';
t=1:k;
for i=1:length(BZ_tensor(1,1,:))
    y = net1(y0);
    y0=y;
    XNN = [XNN; y];
end
XNN=XNN';
XNNreal = Ut * XNN;



%% DMD prediction
H=[];
%r=;
dt= 1;
t = 1:dt:length(BZ_tensor(1,1,:));

point=10;
% for j=1:n
%    H=[H; X(point,j:end-n+j)]; % Focus on the point point: how vary in time
% end 

tend=1100; %it's an index so to have last value to make a cross-validation of my prediction
[PhiH, LambdaH, bH] = DMD(X(:,1:tend-1), X(:,2:tend), r);  % set 'max' to r to get the maximum rank
Lambdadiag=diag(LambdaH);
LambdaHexp=zeros(size(LambdaH));
clear XDMDH
for i= 1:length(t)
    LambdaHexp(1:1+size(LambdaH,1):end) = exp(log(Lambdadiag)/dt*t(i));
    XDMDH(:, i) = PhiH*LambdaHexp*bH;
end



%%
figure (2)
jpred=1200;
subplot(1,3,2)
pcolor(reshape(XNNreal(:,jpred),size(BZ_tensor(:,:,jpred)))); shading interp; colormap(hot)
title('NN','FontSize',40)
set(gca, 'FontSize',35)

caxis([20 140])
subplot(1,3,1)
pcolor(BZ_tensor(:,:,jpred)); shading interp; colormap(hot)
title('Truth','FontSize',40)
caxis([20 140])
set(gca, 'FontSize',35)
subplot(1,3,3)
pcolor(real(reshape(XDMDH(:,jpred),size(BZ_tensor(:,:,jpred))))); shading interp; colormap(hot)
title('Low rank DMD','FontSize',40)
caxis([20 140])
hp4 = get(subplot(1,3,3),'Position');
colorbar('Position', [hp4(1)+hp4(3)+0.01  hp4(2)  0.01  hp4(4)-0.03])
set(gca, 'FontSize',35)

%set(gcf, 'Renderer', 'painters')
%% Save
fileIDa=fopen('Homework/EX4contourpred.dat','w'); % apre file scrittura risultati
fprintf(fileIDa,'VARIABLES=x,y,us,uaprox,unn\n');
fprintf(fileIDa,strcat('ZONE T="',fileIDa,'"\n'));
cont=[length(BZ_tensor(:,1,1));length(BZ_tensor(1,:,1))];
fprintf(fileIDa,'I=%u J=%u F=POINT\n',cont);
write=[reshape([1:length(BZ_tensor(:,1,1))]'.*ones(size(BZ_tensor(:,:,1))),[prod(size(BZ_tensor(:,:,jpred))) 1]),reshape([1:length(BZ_tensor(1,:,1))].*ones(size(BZ_tensor(:,:,1))),[prod(size(BZ_tensor(:,:,1))) 1]),reshape(BZ_tensor(:,:,jpred),[prod(size(BZ_tensor(:,:,1))) 1]),reshape(XDMDH(:,jpred),[prod(size(X_vec_approx(:,:,1))) 1]),reshape(XNNreal(:,jpred),[prod(size(X_vec_approx(:,:,1))) 1])];
fprintf(fileIDa,'%8.6f %8.6f %8.6f %8.6f %8.6f\r\n',write');
clear write
fclose(fileIDa);
clear fileIDa 


%% Neural network reconstruction at point point
% y0 = BZ_tensor(1:limit,1:limit,1);
% for i=1:length(t)
%     y=net(y0);
%     y0=y;
%     y = reshape(y,[1,prod(size(y))]);
%     XNN (1,i) = y(point);
% end
% 
% plot(t,XNN)
% hold on
% plot([t(jmax) t(jmax)],[X(point,jmax)-10 X(point,jmax)+10],'--')

%% SVD low rank reconstruction for prediction
A = S*V';
[PhiH, LambdaH, bH] = DMD(A(:,1:tend-1), A(:,2:tend), r);  % set 'max' to r to get the maximum rank
Lambdadiag=diag(LambdaH);
LambdaHexp=zeros(size(LambdaH));

for i= 1:length(t)
    LambdaHexp(1:1+size(LambdaH,1):end) = exp(log(Lambdadiag)/dt*t(i));
    XDMDHSVD(:, i) = PhiH*LambdaHexp*bH;
end

figure (3)

%% NN on one point
jmax = 1100; % time of training

input=[]; 
output=[];
BZ_vector = reshape(BZ_tensor,[prod(size(BZ_tensor(:,:,1))),k]);

for j=1:jmax  % training trajectories
    input=[input; BZ_vector(point,j)];
    output=[output; BZ_vector(point,j+1)];
%     plot3(y(:,1),y(:,2),y(:,3)), hold on
%     plot3(x0(1),x0(2),x0(3),'ro')
end

%%
net = feedforwardnet([10 20 20 10]);
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net.layers{4}.transferFcn = 'radbas';

%%
net = train(net,input.',output.');

%%
y0 = BZ_vector(point,1);
for i=1:length(t)
    y=net(y0);
    y0=y;
    XNN1 (1,i) = y;
end

%% Figure
figure (3)
patch([t(jmax) t(jmax) t(end) t(end)], [50 110 110 50], 'white','LineStyle','none','FaceColor', 'yellow', 'FaceAlpha', 0.3);
hold on
h(1)=plot(t,X(point,1:length(t)),'LineWidth',6);
hold on
h(2)=plot(t,XNNreal(point,:),'--','LineWidth',6);
hold on
h(3)=plot(t,XDMDH(point,:),'--','LineWidth',6);
hold on
h(4)=plot(t,U(point,1:r)*XDMDHSVD(1:r,:),'--','LineWidth',5);
hold on
h(5)=plot(t,XNN1,':','LineWidth',5);
hold on
text(1150, 55, 'Prediction','FontSize',35,'FontWeight','bold','HorizontalAlignment','center')
legend(h,'Truth','NN small rank','full DMD','small rank DMD','NN on a point','FontSize',35,'Location','Eastoutside')
grid on
set(gca, 'GridLineStyle','--','FontSize',35)
xlabel('Time','FontSize',40,'FontWeight','bold')
ylabel('Value','FontSize',40,'FontWeight','bold')
ylim([50 115]);
set(gcf, 'Renderer', 'painters')

%%
JmaxNN = jmax+5;
y=net(BZ_tensor(1:limit,1:limit,JmaxNN));
figure
subplot(1,2,1)
pcolor(BZ_tensor(1:limit,1:limit,JmaxNN+1)), shading interp
subplot(1,2,2)
pcolor(y), shading interp


%% SINDy
[Theta, H] = poolData(t,1,2,1); % create the library
lambda = 0.025;
Xi = sparsifyDynamics(Theta, X(point,:), lambda, 1);
hold on
plot(t,Theta*Xi)




