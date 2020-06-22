clear all, close all

% Simulate Lorenz system
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; 
r=[10 28 40];

jmax=100; % number of training
x0_vect=30*(rand(3,jmax)-0.5);
input=[]; output=[];

for i=1:length(r)
Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r(i) * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

for j=1:jmax  % training trajectories
    x0 = x0_vect(:,j);
    [t,y] = ode45(Lorenz,t,x0);
    input=[input; [y(1:end-1,:) r(i)*ones(length(y(:,1))-1,1)]];
    output=[output; [y(2:end,:) r(i)*ones(length(y(:,1))-1,1)]];
%     plot3(y(:,1),y(:,2),y(:,3)), hold on
%     plot3(x0(1),x0(2),x0(3),'ro')
end
end
grid on, view(-23,18)


%%
net = feedforwardnet([10 20 20 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'tansig';
net.layers{4}.transferFcn = 'purelin';

%%
net = train(net,input.',output.');

%% Rho trained
figure(2)
x0=30*(rand(3,1)-0.5);
r1=10;
Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r1 * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);    
[t,y] = ode45(Lorenz,t,x0);
h1=plot3(y(:,1),y(:,2),y(:,3));
hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

ynn(1,:)=x0;
for jj=2:length(t)
    y0=net([x0;r1]);
    ynn(jj,1:3)=y0(1:3).'; x0=y0(1:3);
end
h2=plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2]);

figure(3)
subplot(3,3,1), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
title(strcat('\rho=',num2str(r(1))),'FontSize',15)
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontSize',15,'FontWeight','bold')
ylabel('x','FontSize',15,'FontWeight','bold')
subplot(3,3,4), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontSize',15,'FontWeight','bold')
ylabel('y','FontSize',15,'FontWeight','bold')
subplot(3,3,7), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontSize',15,'FontWeight','bold')
ylabel('z','FontSize',15,'FontWeight','bold')


figure(2)
r2=28;
Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r2 * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);    
x0=20*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);
h3=plot3(y(:,1),y(:,2),y(:,3));
hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

ynn(1,:)=x0;
for jj=2:length(t)
    y0=net([x0;r2]);
    ynn(jj,:)=y0(1:3).'; x0=y0(1:3);
end
h4=plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2]);

figure(3)
subplot(3,3,2), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
title(strcat('\rho=',num2str(r(2))),'FontSize',15)
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontSize',15,'FontWeight','bold')
ylabel('x','FontSize',15,'FontWeight','bold')
subplot(3,3,5), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontSize',15,'FontWeight','bold')
ylabel('y','FontSize',15,'FontWeight','bold')
subplot(3,3,8), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontSize',15,'FontWeight','bold')
ylabel('z','FontSize',15,'FontWeight','bold')

figure(2)
xlabel('x','FontSize',15,'FontWeight','bold')
ylabel('y','FontSize',15,'FontWeight','bold')
zlabel('z','FontSize',15,'FontWeight','bold')
r3=40;
Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r3 * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);    
[t,y] = ode45(Lorenz,t,x0);
h5=plot3(y(:,1),y(:,2),y(:,3));
hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

ynn(1,:)=x0;
for jj=2:length(t)
    y0=net([x0;r3]);
    ynn(jj,1:3)=y0(1:3).'; x0=y0(1:3);
end
h6=plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2]);
legend([h1,h2,h3,h4,h5,h6],strcat('ode45 \rho=',num2str(r(1))),strcat('NN \rho=',num2str(r(1))),strcat('ode45 \rho=',num2str(r(2))),strcat('NN \rho=',num2str(r(2))),strcat('ode45 \rho=',num2str(r(3))),strcat('NN \rho=',num2str(r(3))),'FontSize',15,'NumColumns',2)

figure(3)
subplot(3,3,3), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
title(strcat('\rho=',num2str(r(3))),'FontSize',15)
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontSize',15,'FontWeight','bold')
ylabel('x','FontSize',15,'FontWeight','bold')

subplot(3,3,6), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontSize',15,'FontWeight','bold')
ylabel('y','FontSize',15,'FontWeight','bold')

subplot(3,3,9), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontSize',15,'FontWeight','bold')
ylabel('z','FontSize',15,'FontWeight','bold')


%%
figure(2)
x0=30*(rand(3,1)-0.5);
r1=17;
Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r1 * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);    
[t,y] = ode45(Lorenz,t,x0);
h1=plot3(y(:,1),y(:,2),y(:,3));
hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
xlabel('x','FontSize',15,'FontWeight','bold')
ylabel('y','FontSize',15,'FontWeight','bold')
zlabel('z','FontSize',15,'FontWeight','bold')
grid on

ynn(1,:)=x0;
for jj=2:length(t)
    y0=net([x0;r1]);
    ynn(jj,1:3)=y0(1:3).'; x0=y0(1:3);
end
h2=plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2]);

figure(3)
subplot(3,2,1), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
title(strcat('\rho=',num2str(r1)),'FontSize',15)
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontSize',15,'FontWeight','bold')
ylabel('x','FontSize',15,'FontWeight','bold')
subplot(3,2,3), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontSize',15,'FontWeight','bold')
ylabel('y','FontSize',15,'FontWeight','bold')
subplot(3,2,5), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontSize',15,'FontWeight','bold')
ylabel('z','FontSize',15,'FontWeight','bold')


figure(2)
r2=35;
Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r2 * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);    
x0=20*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);
h3=plot3(y(:,1),y(:,2),y(:,3));
hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

ynn(1,:)=x0;
for jj=2:length(t)
    y0=net([x0;r2]);
    ynn(jj,:)=y0(1:3).'; x0=y0(1:3);
end
h4=plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2]);
legend([h1,h2,h3,h4],strcat('ode45 \rho=',num2str(r1)),strcat('NN \rho=',num2str(r1)),strcat('ode45 \rho=',num2str(r2)),strcat('NN \rho=',num2str(r2)),'FontSize',15)

figure(3)
subplot(3,2,2), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
title(strcat('\rho=',num2str(r2)),'FontSize',15)
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontSize',15,'FontWeight','bold')
ylabel('x','FontSize',15,'FontWeight','bold')
subplot(3,2,4), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontSize',15,'FontWeight','bold')
ylabel('y','FontSize',15,'FontWeight','bold')
subplot(3,2,6), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontSize',15,'FontWeight','bold')
ylabel('z','FontSize',15,'FontWeight','bold')

%%
figure(2), view(-75,15)
figure(3)
subplot(3,2,1), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,2), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,3), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,4), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,5), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,6), set(gca,'Fontsize',[15],'Xlim',[0 8])
legend('Lorenz','NN')

%% Attractors points
x1 = [sqrt(b.*(r(2)-1)) sqrt(b.*(r(2)-1)) r(2)-1];
x2 = [-sqrt(b.*(r(2)-1)) -sqrt(b.*(r(2)-1)) r(2)-1];
x3 = [0 0 0];

% Definition of a line passing x3 and perpendicular to line x1-x2
m = - 1./(x2(2)-x1(2)).*(x2(1)-x1(1));

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r(2) * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

x0 = 30*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);


rightLine = (y(:,2)>m*y(:,1));
leftLine = ~rightLine;

yr = y;
yl = y;

yr(~rightLine,:) = NaN;
yl(rightLine,:) = NaN;

figure
fill3([max(y(:,1)) min(y(:,1)) min(y(:,1)) max(y(:,1))], [-max(y(:,1)) -min(y(:,1)) -min(y(:,1)) -max(y(:,1))], [max(y(:,3)) max(y(:,3)) 0 0], 'g')
hold on
plot3(y(:,1),y(:,2),y(:,3),'r','LineWidth',2)
hold on
plot3(yr(:,1),yr(:,2),yr(:,3),'r',yl(:,1),yl(:,2),yl(:,3),'b','LineWidth',2)
alpha(0.3)
grid on, view(-23,18)
hold on
plot3([x1(1) x2(1) x3(1)], [x1(2) x2(2) x3(2)], [x1(3) x2(3) x3(3)],'o','LineWidth',2)
text([x1(1) x2(1) x3(1)], [x1(2) x2(2) x3(2)], [x1(3) x2(3) x3(3)],{'Attractor 1','Attractor 2','Origin'},'VerticalAlignment','bottom','HorizontalAlignment','right','FontSize',15,'FontWeight','bold')
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('x','FontSize',15,'FontWeight','bold')
ylabel('y','FontSize',15,'FontWeight','bold')
zlabel('z','FontSize',15,'FontWeight','bold')


%plot3(y(:,1),y(:,2),y(:,3))

clas_vector=zeros(length(rightLine),1);
clas_vector(rightLine==1) = 1;
clas_vector(rightLine==0) = -1;

jump = abs(clas_vector(1:end-1)-clas_vector(2:end));
figure
plot(jump)

jmax=200; % number of training
x0_vect=30*(rand(3,jmax)-0.5);

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r(2) * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);
%%

net=lobeNN(jmax,x0_vect,t,2,Lorenz);
net5=lobeNN(jmax,x0_vect,t,6,Lorenz);
net10=lobeNN(jmax,x0_vect,t,11,Lorenz);
net20=lobeNN(jmax,x0_vect,t,21,Lorenz);
net30=lobeNN(jmax,x0_vect,t,31,Lorenz);
net40=lobeNN(jmax,x0_vect,t,41,Lorenz);

%%
for i=1:50
x0 = 30*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);
rightLine = (y(:,2)>m*y(:,1));
leftLine = ~rightLine;

clas_vector=zeros(length(rightLine),1);
clas_vector(rightLine==1) = 1;
clas_vector(rightLine==0) = -1;
jumpcalc = abs(clas_vector(1:end-1)-clas_vector(2:end))/2;


ynn(1,:)=x0;
for jj=1:length(t)-1
    y0 = net(y(jj,:)');
    jumpNN(jj+1)=y0.';

    if jj<length(t)-4
    y05 = net5(y(jj,:)');
    jumpNN5(jj+5)=y05.';
    end
    if jj<length(t)-9
    y10 = net10(y(jj,:)');
    jumpNN10(jj+10)=y10.';
    end
    if jj<length(t)-19
    y20 = net20(y(jj,:)');
    jumpNN20(jj+20)=y20.';
    end
    if jj<length(t)-29
    y30 = net30(y(jj,:)');
    jumpNN30(jj+30)=y30.';
    end
    if jj<length(t)-39
    y40 = net40(y(jj,:)');
    jumpNN40(jj+40)=y40.';
    end
end
err(i,:)=[sum(round(max((jumpcalc'-jumpNN(1:end-1)),0))) sum(round(max(jumpcalc'-jumpNN5(1:end-1),0))) sum(round(max(jumpcalc'-jumpNN10(1:end-1),0))) sum(round(max(jumpcalc'-jumpNN20(1:end-1),0))) sum(round(max((jumpcalc'-jumpNN30(1:end-1)),0))) sum(round(max(jumpcalc'-jumpNN40(1:end-1),0)))];
end

% figure (4)
% plot(jumpcalc)
% hold on
% plot(round(jumpNN),'--')
% hold on
% plot(round(jumpNN5),'--')
% hold on
% plot(round(jumpNN10),'--')
% hold on
% plot(round(jumpNN15),'--')

figure
bar(1:length(jumpcalc),[jumpcalc'; round(jumpNN(1:end-1)); round(jumpNN5(1:end-1)); round(jumpNN10(1:end-1)); round(jumpNN20(1:end-1)); round(jumpNN30(1:end-1)); round(jumpNN40(1:end-1))],5,'stacked')
legend('ode45','NN 1','NN 5','NN 10','NN 20','NN 30','NN 40','FontSize',15)
grid on
set(gca, 'GridLineStyle','--','FontSize',13)
xlabel('Time','FontSize',15,'FontWeight','bold')
set(gca,'YTickLabel',[]);

figure
bar( mean(err,1),'b')
ylabel('Avarage error','FontSize',15,'FontWeight','bold')
xlabel('\Deltat','FontSize',15,'FontWeight','bold')
grid on
set(gca, 'GridLineStyle','--')
set(gca,'xticklabel',{'1', '5', '10', '20', '30', '40'},'FontSize',15)



