%% Exercise 1
clear all
clc

%% Import input files
data = importfile("Ex 1 input.txt"); % columns are year, Snowshow Hare Pelts and	Canada Lynx Pelts
% figure (1)
% plot(data(:,1),data(:,2))
% hold on
% plot(data(:,1),data(:,3))

%% Construction of matrixes to aplly DMD
X = [data(:,2)'.*10^3; data(:,3)'.*10^3];
t = (data(:,1) - data(1,1));
dt = t(2) - t(1);3

%% DMD architecture
[Phi, Lambda, b, V] = DMD(X(:,1:end-1), X(:,2:end), 'max');

XDMD = zeros(2, length(t));
for i= 1:length(t)
    XDMD(:, i) = Phi*[exp(log(Lambda(1,1))/dt*t(i)) 0; 0 exp(log(Lambda(2,2))/dt*t(i))]*b;
end

%% Time delay DMD
H=[];
n=7; % numbers of time delay
X1=X(:,1:end-2);
for j=1:n
  H=[H; X1(:,j:end-n+j)]; 
end 
[PhiH, LambdaH, bH, Vh] = DMD(H(:,1:end-1), H(:,2:end), 'max');  % set 'max' to r to get the maximum rank
XDMDH = zeros(length(H(:,1)), length(t));
Lambdadiag=diag(LambdaH);
LambdaHexp=zeros(size(LambdaH));

t1 = t; %[t; t(end)+dt; t(end)+2*dt];
for i= 1:length(t1)
    LambdaHexp(1:1+size(LambdaH,1):end) = exp(log(Lambdadiag)/dt*t1(i));
    XDMDH(:, i) = PhiH*LambdaHexp*bH;
end

%%
figure (1)
xBox = [data(1,1)+length(X1(1,:))*2, data(1,1)+length(X1(1,:))*2, t(end)+data(1,1), t(end)+data(1,1), data(1,1)+length(X1(1,:))*2];
yBox = [-5*10^4, 20*10^4, 20*10^4, -5*10^4, -5*10^4];
patch(xBox, yBox, 'white','LineStyle','none','FaceColor', 'yellow', 'FaceAlpha', 0.3);
hold on
h(1)=plot(t(:,1)+data(1,1),X(1,:),'LineWidth',2);
hold on
h(2)=plot(t(:,1)+data(1,1),X(2,:),'LineWidth',2);
hold on
h(3)=plot(t(:,1)+data(1,1),XDMD(1,:),':','LineWidth',2);
hold on
h(4)=plot(t(:,1)+data(1,1),XDMD(2,:),':','LineWidth',2);
hold on
h(5)=plot(t1+data(1,1),XDMDH(1,:),'--','LineWidth',2);
hold on
h(6)=plot(t1+data(1,1),XDMDH(2,:),'--','LineWidth',2);
hold on
grid on
set(gca, 'GridLineStyle','--','FontSize',15)
ylabel('Population','FontSize',18,'FontWeight','bold')
xlabel('Year','FontSize',18,'FontWeight','bold')
xlim([data(1,1) t1(end)+data(1,1)])
xline(data(1,1)+length(X1(1,:))*2)
hold off
ylim([0 16*10^4])
text(data(1,1)+length(X1(1,:))*2, max(data(:,2))*10^3-10^4, {'Prediction'},'FontWeight','bold','HorizontalAlignment','center','FontSize',18)
legend(h,'Snowshow Hare (SH)','Canada Lynx (CL)','SH DMD','CL DMD','SH DMD delayed','CL DMD delayed','FontSize',15,'Location','Northoutside','NumColumns',3)
set(gcf, 'Renderer', 'painters')

%% Plot of eigenvalues
figure (3)
plot(real(log(diag(Lambda))),imag(log(diag(Lambda))), 'o', 'LineWidth', 2)
hold on
plot(real(log(Lambdadiag)),imag(log(Lambdadiag)), 'o', 'LineWidth', 2)
set(gca, 'YAxisLocation', 'origin', 'XAxisLocation', 'origin', 'GridLineStyle','--','FontSize',15)
xlabel('Re','FontSize',18,'FontWeight','bold')
ylabel('Im','FontSize',18,'FontWeight','bold')
legend('DMD','Time-delay DMD','FontSize',15,'Location','NorthWest')
grid on

%% Plot eigenvectors
figure (4)
subplot(2,1,1), plot(0:1:length(V(:,1))-1, V,'Linewidth',2)
set(gca, 'GridLineStyle','--','FontSize',15)
legend('Mode 1','Mode 2','FontSize',15,'Location','NorthEast')
title('DMD','FontSize',18,'FontWeight','bold')
grid on
xlim([0 length(V(:,1))-1])

subplot(2,1,2), plot(0:1:length(Vh(:,1))-1, Vh(:,1:4),'Linewidth',2)
set(gca, 'GridLineStyle','--','FontSize',15)
legend('Mode 1','Mode 2','Mode 3','Mode 4','FontSize',15,'Location','NorthEast')
title('Time-delay DMD','FontSize',18,'FontWeight','bold')
grid on
xlim([0 length(V(:,1))-1])

%% Empirical predator-pray models
% xdot = (b - py)x
% ydot = (rx - d)y
% Find the coefficients b, p, r and d
x = X(1,:);
y = X(2,:);


b=0.55;
p=1.5e-05;
r=1e-05;
d=0.49;
Xdot = @(t,z)([ (b-p*z(2))*z(1)       ; ...
                (r*z(1)-d)*z(2)]);             
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);
[t,XV] = ode45(Xdot,t,[data(1,2)*10^3 data(1,3)*10^3]);

figure
plot(t(:,1)+data(1,1),X(1,:),'LineWidth',2);
hold on
plot(t(:,1)+data(1,1),X(2,:),'LineWidth',2);
hold on
plot(t(:,1)+data(1,1),XV(:,1),'--','LineWidth',2)
hold on 
plot(t(:,1)+data(1,1),XV(:,2),'--','LineWidth',2)
hold on
ylabel('Population','FontSize',18,'FontWeight','bold')
xlabel('Year','FontSize',18,'FontWeight','bold')
legend('Snowshow Hare (SH)','Canada Lynx (CL)','SH Lotka-Volterra','CL Lotka-Volterra','FontSize',15,'Location','Northoutside','NumColumns',3)
ylim([0 16*10^4])
xlim([1845 1903])
set(gca, 'GridLineStyle','--','FontSize',15)
grid on
norm(XV(:,1)'-X(1,:))/norm(X(1,:))
norm(XV(:,2)'-X(2,:))/norm(X(2,:))


%% Application of Sindy to find best non-linear fit Snowshow
[Theta, H] = poolData(t,1,4,1); % create the library
lambda = 0.025;
Xi = sparsifyDynamics(Theta, X(1,:)', lambda, 1);


%% Application of Sindy to find best non-linear fit Lynx
[Theta2, H2] = poolData(t,1,4,1); % create the library
lambda = 0.025;
Xi2 = sparsifyDynamics(Theta2, X(2,:)', lambda, 1);

figure (3)
plot(data(:,1),X(1,:),'LineWidth',2)
hold on
plot(data(:,1),X(2,:),'LineWidth',2)
hold on
plot(data(:,1),Theta*Xi,'--','LineWidth',2)
hold on
plot(data(:,1),Theta2*Xi2,'--','LineWidth',2)

set(gca, 'GridLineStyle','--','FontSize',15)
ylabel('Population','FontSize',18,'FontWeight','bold')
xlabel('Year','FontSize',18,'FontWeight','bold')
xlim([data(1,1) t1(end)+data(1,1)])
grid on
legend('Snowshow Hare (SH)','Canada Lynx (CL)','SH SINDy','CL SINDy','FontSize',15,'Location','Eastoutside','numcolumns',2)

norm(Theta*Xi-X(1,:)')/norm(X(1,:))
norm(Theta2*Xi2-X(2,:)')/norm(X(2,:))

%%
figure 
b = bar([Xi Xi2],0.9,'FaceColor','flat');
set(gca,'xticklabel',string(H),'FontSize',25)
ax=gca;
ax.XTick = [1:length(Xi)];
xtickangle(90)
ylabel('Weight','FontWeight','bold')
set(gca, 'GridLineStyle','--','FontSize',25)
grid on
legend('Snowshow Hare','Canada Lynx','FontSize',25,'Location','NorthEast')

