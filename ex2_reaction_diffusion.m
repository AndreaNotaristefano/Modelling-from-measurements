clear all 
clc
load('reaction_diffusion_big.mat')

%% SVD x direction
u_vec=[];
N=length(x)*length(y);
for i=1:length(u(1,1,:))
    u_vec=[u_vec; reshape(u(:,:,i),[1,N])];
end
u_vec=u_vec';
[Uu, Su, Vu] = svd(u_vec,'econ');

%% Truncation
r = 6;
U=Uu(:,1:r); 
S=Su(1:r,1:r); 
V=Vu(:,1:r);

%% Check the SVD truncation outcome
u_approx = Uu(:,1:r)*Su(1:r,1:r)*Vu(:,1:r)';
u_approx= u_approx';
for i=1:length(u(1,1,:))
    u_vec_approx(:,:,i)=reshape(u_approx(i,:),[length(x),length(y)]);
end
figure
subplot(2,2,[1,2])
plot(diag(Su)/sum(diag(Su)),'*b')
set(gca, 'GridLineStyle','--','FontSize',13)
xlim([0 20])
title('Modes','FontSize',15)
grid on

diagonal=diag(Su)/sum(diag(Su));
us=u(:,:,1)';
uaprsave=u_vec_approx(:,:,1)';

fileIDa=fopen('Homework/RD.dat','w'); % apre file scrittura risultati
fprintf(fileIDa,'VARIABLES=sigma,x,y,us,uaprox\n');
fprintf(fileIDa,strcat('ZONE T="',fileIDa,'"\n'));
cont=[length(x);length(y)];
fprintf(fileIDa,'I=%u J=%u F=POINT\n',cont);
write=[reshape(x'.*ones(size(us)),[prod(size(us)) 1]),reshape((y'.*ones(size(us)))',[prod(size(us)) 1]),reshape(us,[prod(size(us)) 1]),reshape(uaprsave,[prod(size(us)) 1])];
fprintf(fileIDa,'%8.6f %8.6f %8.6f %8.6f\r\n',write');
clear write
fclose(fileIDa);
clear fileIDa 


subplot(2,2,3)
pcolor(x,y,u(:,:,1)); shading interp;
colorbar
set(gca,'FontSize',13)
title('Truth','FontSize',15)
caxis([-1 1])

subplot(2,2,4)
pcolor(x,y,u_vec_approx(:,:,1)); shading interp;
colorbar
set(gca,'FontSize',13)
title('r=6 approximation','FontSize',15)
caxis([-1 1])


%% Prediction
clear XDMDH
H=[];
%n=10; % numbers of time delay
%r=;
dt= t(2)-t(1);
point=10;
% for j=1:n
%   H=[H; u_vec(j:n+j-1,point)']; % Focus on the first point: how vary in time
% end 
A = S*V';
tend=181; %it's an index so to have last value to make a cross-validation of my prediction
[PhiH, LambdaH, bH] = DMD(A(:,1:tend-1), A(:,2:tend), 'max');  % set 'max' to r to get the maximum rank
Lambdadiag=diag(LambdaH);
LambdaHexp=zeros(size(LambdaH));

for i= 1:length(t)
    LambdaHexp(1:1+size(LambdaH,1):end) = exp(log(Lambdadiag)/dt*t(i));
    XDMDH(:, i) = PhiH*LambdaHexp*bH;
end
figure
plot(t,u_vec(point,1:length(t)),'LineWidth',3)
hold on
plot(t,Uu(point,1:r)*XDMDH(1:r,:),'--','LineWidth',3)
hold on
plot([t(tend) t(tend)],[u_vec(point,tend)-1 u_vec(point,tend)+1],'--','LineWidth',2)
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontWeight','bold','FontSize',15);
ylabel('Value','FontWeight','bold','FontSize',15);
grid on

legend('Truth','r=6 approximation','Start prediction','FontSize',15,'Location','NorthWest')

figure
subplot(1,2,1)
pcolor(x,y,u(:,:,end)); shading interp;
set(gca, 'GridLineStyle','--','FontSize',13)
colorbar
title('Truth','FontSize',15)
caxis([-1 1])

subplot(1,2,2)
u_predict=real(reshape(Uu(:,1:r)*XDMDH(1:r,:),size(u)));
pcolor(x,y,u_predict(:,:,end)); shading interp;
colorbar
set(gca, 'GridLineStyle','--','FontSize',13)
title('r=6 approximation','FontSize',15)
caxis([-1 1])

fileIDa=fopen('Homework/RDprediction.dat','w'); % apre file scrittura risultati
fprintf(fileIDa,'VARIABLES=sigma,x,y,us,uaprox\n');
fprintf(fileIDa,strcat('ZONE T="',fileIDa,'"\n'));
cont=[length(x);length(y)];
fprintf(fileIDa,'I=%u J=%u F=POINT\n',cont);
write=[reshape(x'.*ones(size(us)),[prod(size(us)) 1]),reshape((y'.*ones(size(us)))',[prod(size(us)) 1]),reshape(u(:,:,end),[prod(size(us)) 1]),reshape(u_predict(:,:,end),[prod(size(us)) 1])];
fprintf(fileIDa,'%8.6f %8.6f %8.6f %8.6f\r\n',write');
clear write
fclose(fileIDa);
clear fileIDa 
