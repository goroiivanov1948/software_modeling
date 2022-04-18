
%% Входные данные 
clear;
clc;
T_imp = 16; % длина прямоугольных импульсов полезного сигнала
T_int_imp = T_imp*4; % длина интервала прямоугольного импульса
T_okn = T_int_imp*4; % длина окна (промежуток стабилизации)
N_iter = 2500; % количество итераций для цифровой обработки
T = N_iter*T_okn; % длительность полученного сигнала
A_sign = 1; % амплитуда полезного сигнала (прямоугольного импульса)
A_pom = 1; % амплитуда сигнала узкополосной помехи
F0=0; % частота прямоугольного импульса
F_pom = 10; % частота сигнала узкополосной помехи
A_sim = [1 0]; % амплитуда передаваемого символа 
No_iter = 100; % Номер текущего такта обработки сигнала
%No_iter = input('Введите номер такта в пределах от 1 до N_iter = 2500: ');
n_period = 1; % Количество периодов отображения на графике

%% Функция генерации прямоугольного импульса для передачи 2-х бит информации(1 символа)

M_sign_no_isk = zeros(T,1);

for k = 1:N_iter
        for i = 1:T_okn
            if (A_sim(1) == 0 && A_sim(2) == 0)
                n_simv = 1;
            elseif (A_sim(1) == 0 && A_sim(2) == 1)
                n_simv = 2;
            elseif (A_sim(1) == 1 && A_sim(2) == 0)
                n_simv = 3;
            elseif (A_sim(1) == 1 && A_sim(2) == 1)
                n_simv = 4; 
            end
            if (i > (n_simv*T_int_imp-T_int_imp/2-T_imp/2) && i <=...
                    (n_simv*T_int_imp-T_int_imp/2+T_imp/2))
                M_sign_no_isk(i+(k-1)*T_okn) = A_sign*exp(2*pi*i*1i*F0/2500);
            end
        end
end

% Расчет мощности полезного сигнала

P_M_sign_no_isk = 0;
for i = 1 : n_period*T_okn
    P_M_sign_no_isk = P_M_sign_no_isk + power(abs(M_sign_no_isk((No_iter-1)*T_okn+i)), 2);
end
P_M_sign_no_isk = P_M_sign_no_isk/T_okn;
str = num2str(P_M_sign_no_isk);
%title(['Ps` = ' str],'interpreter','latex');

figure(1);
plot(real(M_sign_no_isk),'b');
hold on;
plot(imag(M_sign_no_isk),'r--');
legend('Real', 'Imaginary');

axis([(No_iter-1)*T_okn,(No_iter-1)*T_okn+n_period*T_okn,...
        -0.5*max(real(M_sign_no_isk((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))),...
        max(real(M_sign_no_isk((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
        0.2*max(real(M_sign_no_isk((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))]);

grid on;

set(gca,'Xtick',(No_iter-1)*T_okn : T_okn/4 : (No_iter-1)*T_okn+n_period*T_okn);

set(gca,'Ytick',-0.5*max(real(M_sign_no_isk((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))):...
                0.1:...
                max(real(M_sign_no_isk((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
                0.2*max(real(M_sign_no_isk((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))));

title("Неискаженный импульс полезного сигнала мощностью",[' Ps` = ' str],'interpreter','latex');
%subtitle([' Ps` = ' str],'interpreter','latex')
xlabel('n, номер отсчета');
ylabel('U');

%% Функция генерации сигнала N-ого количества активных узкополосных помех

No_iter = 1000;
M_pom = zeros(T,1);

Number_pom = 3; % Число помех
M_all_pom = zeros(T, Number_pom); % Массив для N-oго количества помех
Q_isk_pom_all = zeros(N_iter, Number_pom); % Начальная фаза помеховых сигналов
F = zeros(Number_pom, 1); % массив для частот помеховых сигналов
A_rand = zeros(Number_pom, 1); % Массив для амплитуд, заполненный нулями

for i = 1:Number_pom
     Q_isk_pom_all(1:N_iter, Number_pom) = unifrnd(0,0.2,N_iter,1); % Цикл для генерации фазого сдвига
end

F(1:3) = [28 15 20]; % Частота действующих в канале связи узкополосных помех

for j = 1:Number_pom
    A_rand(j) = A_pom; % Амплитуда помеховых сигналов
    for i = 1:T
        M_all_pom(i, j) = A_rand(j)*exp(2*pi*1i*i*F(j)/T_okn); % генерация сигналов помех от 1 до N (1,2,3,...)
    end
end


for i = 1:Number_pom
    M_pom(1:T) = M_pom(1:T) + M_all_pom(1:T, i); % Суммарное значение действующих сигналов помехи
end

% Расчет мощности сигнала узкополосных помех
P_M_pom = 0;
for i = 1 : n_period*T_okn
    P_M_pom = P_M_pom + power(abs(M_pom((No_iter-1)*T_okn+i)), 2);
end
P_M_pom = P_M_pom/T_okn;
str_1 = num2str(P_M_pom);
n = num2str(Number_pom);

figure(2);
%subplot(211)
plot(real(M_pom),'b');
hold on;
plot(imag(M_pom),'r--');

title({"Сигнал узкополосных помех мощностью",[' Pi` = ' str_1], ['при количестве активных помех n = ' n]});   
legend('Real', 'Imaginary');

axis([(No_iter-1)*T_okn,(No_iter-1)*T_okn+n_period*T_okn,...
        -1.5*max(real(M_pom((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))),...
        max(real(M_pom((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
        0.5*max(real(M_pom((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))]);

grid on;

set(gca,'Xtick',(No_iter-1)*T_okn : T_okn/4 : (No_iter-1)*T_okn+n_period*T_okn);

set(gca,'Ytick',-1.5*max(real(M_pom((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))):...
                0.4:...
                max(real(M_pom((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
                0.5*max(real(M_pom((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))));

xlabel('n, номер отсчета');
ylabel('U');

%% Функция генерации случайного шума

M_noise = 0.001*wgn(T,1,0, 'complex');
%M_noise = M_noise/max(real(M_noise));

% Расчет мощности белого гауссова шума
P_M_noise = 0;
for i = 1 : n_period*T_okn
    P_M_noise = P_M_noise + power(abs(M_noise((No_iter-1)*T_okn+i)), 2);
end
P_M_noise = P_M_noise/T_okn;
str_2 = num2str(P_M_noise);

%subplot(3,1,3);
figure(3);
subplot(211)
plot(real(M_noise),'b','LineWidth',1);
hold on;
plot(imag(M_noise),'r--');
title({'Сгенерированный белый шум мощностью',[' Pg` = ' str_2]});
legend('Real', 'Imaginary');

axis([(No_iter-1)*T_okn,(No_iter-1)*T_okn+n_period*T_okn,...
        -1.6*max(real(M_noise((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))),...
        max(real(M_noise((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
        0.6*max(real(M_noise((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))]);

grid on;

set(gca,'Xtick',(No_iter-1)*T_okn : T_okn/4 : (No_iter-1)*T_okn+n_period*T_okn);

set(gca,'Ytick',-1.6*max(real(M_noise((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))):...
                0.0008:...
                max(real(M_noise((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
                0.6*max(real(M_noise((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))));
            
%% Полученный в результате (Сигнал + помеха + шум)    

M_Signal = M_sign_no_isk + M_pom + M_noise; % Импульс + помеха + шум

% Отношение сигнал/помеха+шум до фильтра

Q_Signal_Pom_Shum = 10 * log10(P_M_sign_no_isk / (P_M_pom +P_M_noise));
str_3 = num2str(Q_Signal_Pom_Shum);          

figure(4);
%subplot(2,1,1)
plot(real(M_Signal),'b');
hold on;
plot(imag(M_Signal),'r--');
title({'Полученный на вход фильтра сигнал при значении',['SNR = ' str_3 ' dB']});
legend('Real', 'Imaginary');

axis([(No_iter-1)*T_okn,(No_iter-1)*T_okn+n_period*T_okn,...
        -1.2*max(real(M_Signal((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))),...
        max(real(M_Signal((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
        0.3*max(real(M_Signal((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))]);

grid on;

set(gca,'Xtick',(No_iter-1)*T_okn : T_okn/2 : (No_iter-1)*T_okn+n_period*T_okn);

set(gca,'Ytick',-1.2*max(real(M_Signal((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))):...
                0.4:...
                max(real(M_Signal((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
                0.3*max(real(M_Signal((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))));
 
%% Модель фильтра
clc;

No_iter = 1000;
str_4 = num2str(No_iter);

R = eye(T_okn); % корреляционная матрица (в начальном приближении равна единичной матрице)
W = eye(T_okn); % матрица весовых коэффициентов (в начальном приближении равна единичной матрице)
H = zeros(T,1); 
H(1:T_okn) = M_sign_no_isk(1:T_okn,1); % Импульсная характеристика(в начальном приближении равна неискаженному сигналу)
a = 0.005; % коэффициент адаптации

M_pom_shum = M_pom+M_noise; % Суммарное значение помех и белого гауссова шума

for k = 2:N_iter
     
    U = W * M_pom_shum((k-2)*T_okn+1:(k-1)*T_okn);
    Ucs = transpose(conj(U)); % Транспонирование и комплексное сопряжение
    
    h = (W*(R*H((k-2)*T_okn+1:(k-1)*T_okn)))/...
        (sqrt(dot((R*H((k-2)*T_okn+1:(k-1)*T_okn)),(H((k-2)*T_okn+1:(k-1)*T_okn))))*T_okn);
    H((k-1)*T_okn+1:k*T_okn) = h / max(real(h)); % Нормировка в пределах 1
    
    W = 1/(1-a) * ( W - ( a*U*Ucs) /...
        (1 - a + a*(dot(M_pom_shum((k-2)*T_okn+1:(k-1)*T_okn),(U)))));
    
    R = (1-a)*R+a*(((M_Signal((k-2)*T_okn+1:(k-1)*T_okn)))*((M_Signal((k-2) *T_okn+1:(k-1)*T_okn)))');

end


figure
plot(real(H),'b')
hold on;
plot(imag(H),'r')
grid on;

axis([(No_iter-1)*T_okn,(No_iter-1)*T_okn+n_period*T_okn,...
        -0.5*max(real(H((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))),...
        max(real(H((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
        0.2*max(real(H((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))]);

grid on;

set(gca,'Xtick',(No_iter-1)*T_okn : T_okn/4 : (No_iter-1)*T_okn+n_period*T_okn);

set(gca,'Ytick',-0.5*max(real(H((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))):...
                0.2:...
                max(real(H((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
                0.2*max(real(H((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))));

title(['Импульсная характеристика на ', str_4, '-ой итерации']);
xlabel('n, номер отсчета');
ylabel('U');

%% Нахождение свертки через встроенную функцию "conv"
No_iter = 1000;
str_5 = num2str(No_iter);

y = zeros((2*T_okn-1)*N_iter, 1);

for k = 2:N_iter
    v = M_Signal((k-1)*T_okn+1:k*T_okn); % Результирующий сигнал, полученный на вход адаптивного согласованного фильтра
    b = H((k-2)*T_okn+1:(k-1)*T_okn); % Импульсная характеристика системы
    w = conv(v, b); % Свертка сигнала и импульсной характеристики системы
    y((k-1)*(2*T_okn-1)+1 : k*(2*T_okn-1)) = w / (max(real(w))); % Нормировка в пределах 1
end


figure
subplot(211);
plot((real(y)));

axis([(No_iter-1)*2*T_okn,(No_iter-1)*2*T_okn+n_period*2*T_okn-1,...
        -1.2*max(real(y((No_iter-1)*2*T_okn+1:(No_iter-1)*2*T_okn+n_period*2*T_okn-1))),...
        max(real(y((No_iter-1)*2*T_okn+1:(No_iter-1)*2*T_okn+n_period*2*T_okn-1)))+...
        0.5*max(real(y((No_iter-1)*2*T_okn+1:(No_iter-1)*2*T_okn+n_period*2*T_okn-1)))]);

grid on;

set(gca,'Xtick',(No_iter-1)*2*T_okn : T_okn/4 : (No_iter-1)*2*T_okn+n_period*2*T_okn-1);

set(gca,'Ytick',-1.2*max(real(y((No_iter-1)*2*T_okn+1:(No_iter-1)*2*T_okn+n_period*2*T_okn-1))):...
                0.2:...
                max(real(y((No_iter-1)*2*T_okn+1:(No_iter-1)*2*T_okn+n_period*2*T_okn-1)))+...
                0.5*max(real(y((No_iter-1)*2*T_okn+1:(No_iter-1)*2*T_okn+n_period*2*T_okn-1))));

title({['Свертка сигнала и импульсного отклика на ', str_5, '-ой итерации'], ['при количестве активных помех n = ' n]})
xlabel('n, номер отсчета');
ylabel('U');

%% Графики, построенные на одном поле

figure
subplot(321)
plot(real(M_sign_no_isk),'b');
hold on;
plot(imag(M_sign_no_isk),'r--');
legend('Real', 'Imaginary');

axis([(No_iter-1)*T_okn,(No_iter-1)*T_okn+n_period*T_okn,...
        -0.5*max(real(M_sign_no_isk((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))),...
        max(real(M_sign_no_isk((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
        0.2*max(real(M_sign_no_isk((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))]);

grid on;

set(gca,'Xtick',(No_iter-1)*T_okn : T_okn/4 : (No_iter-1)*T_okn+n_period*T_okn);

set(gca,'Ytick',-0.5*max(real(M_sign_no_isk((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))):...
                0.1:...
                max(real(M_sign_no_isk((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
                0.2*max(real(M_sign_no_isk((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))));

title('Неискаженный импульс полезного сигнала мощностью',[' Pg` = ' str]);
xlabel('n, номер отсчета');
ylabel('U');

subplot(322)

plot(real(M_noise),'b','LineWidth',1);
hold on;
plot(imag(M_noise),'r--');
title({'Сгенерированный белый шум мощностью',[' Pg` = ' str_2]});
legend('Real', 'Imaginary');

axis([(No_iter-1)*T_okn,(No_iter-1)*T_okn+n_period*T_okn,...
        -1.6*max(real(M_noise((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))),...
        max(real(M_noise((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
        0.6*max(real(M_noise((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))]);

grid on;

set(gca,'Xtick',(No_iter-1)*T_okn : T_okn/4 : (No_iter-1)*T_okn+n_period*T_okn);

set(gca,'Ytick',-1.6*max(real(M_noise((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))):...
                0.0008:...
                max(real(M_noise((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
                0.6*max(real(M_noise((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))));

subplot(323)

plot(real(M_pom),'b');
hold on;
plot(imag(M_pom),'r--');

title({"Сигнал узкополосных помех мощностью",[' Pi` = ' str_1], ['при количестве активных помех n = ' n]});   
legend('Real', 'Imaginary');

axis([(No_iter-1)*T_okn,(No_iter-1)*T_okn+n_period*T_okn,...
        -1.5*max(real(M_pom((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))),...
        max(real(M_pom((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
        0.5*max(real(M_pom((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))]);

grid on;

set(gca,'Xtick',(No_iter-1)*T_okn : T_okn/4 : (No_iter-1)*T_okn+n_period*T_okn);

set(gca,'Ytick',-1.5*max(real(M_pom((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))):...
                0.4:...
                max(real(M_pom((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
                0.5*max(real(M_pom((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))));

xlabel('n, номер отсчета');
ylabel('U');


subplot(324)
plot(real(M_Signal),'b');
hold on;
plot(imag(M_Signal),'r--');
title({'Полученный на вход фильтра сигнал при значении',['SNR = ' str_3 ' dB']});
legend('Real', 'Imaginary');

axis([(No_iter-1)*T_okn,(No_iter-1)*T_okn+n_period*T_okn,...
        -1.2*max(real(M_Signal((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))),...
        max(real(M_Signal((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
        0.3*max(real(M_Signal((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))]);

grid on;

set(gca,'Xtick',(No_iter-1)*T_okn : T_okn/2 : (No_iter-1)*T_okn+n_period*T_okn);

set(gca,'Ytick',-1.2*max(real(M_Signal((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))):...
                0.4:...
                max(real(M_Signal((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
                0.3*max(real(M_Signal((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))));
            
subplot(325)
plot(real(H),'b')
hold on;
plot(imag(H),'r')
grid on;

axis([(No_iter-1)*T_okn,(No_iter-1)*T_okn+n_period*T_okn,...
        -0.5*max(real(H((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))),...
        max(real(H((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
        0.2*max(real(H((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))]);

grid on;

set(gca,'Xtick',(No_iter-1)*T_okn : T_okn/4 : (No_iter-1)*T_okn+n_period*T_okn);

set(gca,'Ytick',-0.5*max(real(H((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))):...
                0.2:...
                max(real(H((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn)))+...
                0.2*max(real(H((No_iter-1)*T_okn+1:(No_iter-1)*T_okn+n_period*T_okn))));

title(['Импульсная характеристика на ', str_4, '-ой итерации']);
xlabel('n, номер отсчета');
ylabel('U');


subplot(326)
plot((real(y)));

axis([(No_iter-1)*2*T_okn,(No_iter-1)*2*T_okn+n_period*2*T_okn-1,...
        -1.2*max(real(y((No_iter-1)*2*T_okn+1:(No_iter-1)*2*T_okn+n_period*2*T_okn-1))),...
        max(real(y((No_iter-1)*2*T_okn+1:(No_iter-1)*2*T_okn+n_period*2*T_okn-1)))+...
        0.5*max(real(y((No_iter-1)*2*T_okn+1:(No_iter-1)*2*T_okn+n_period*2*T_okn-1)))]);

grid on;

set(gca,'Xtick',(No_iter-1)*2*T_okn : T_okn/4 : (No_iter-1)*2*T_okn+n_period*2*T_okn-1);

set(gca,'Ytick',-1.2*max(real(y((No_iter-1)*2*T_okn+1:(No_iter-1)*2*T_okn+n_period*2*T_okn-1))):...
                0.2:...
                max(real(y((No_iter-1)*2*T_okn+1:(No_iter-1)*2*T_okn+n_period*2*T_okn-1)))+...
                0.5*max(real(y((No_iter-1)*2*T_okn+1:(No_iter-1)*2*T_okn+n_period*2*T_okn-1))));

title({['Свертка сигнала и импульсного отклика на ', str_5, '-ой итерации'], ['при количестве активных помех n = ' n]})
xlabel('n, номер отсчета');
ylabel('U');

%% Расчет мощности неискаженного сигнала

P_M_sign_no_isk = 0;
for i = 1 : n_period*T_okn
    P_M_sign_no_isk = P_M_sign_no_isk + power(abs(M_sign_no_isk((No_iter-1)*T_okn+i)), 2);
end
P_M_sign_no_isk = P_M_sign_no_isk/T_okn;
str = num2str(P_M_sign_no_isk);
title(['Ps` = ' str],'interpreter','latex');

%% Отношение сигнал/помеха+шум до фильтра

Q_Signal_Pom_Shum = 10 * log10(P_M_sign_no_isk / (P_M_pom +P_M_noise));
str = num2str(Q_Signal_Pom_Shum);          
title(['SNR = ' str ' dB'],'interpreter','latex');

%% Построение спектра сигнала и белого шума
Fs = 2500; % Частота дискретизации в Гц (Исследование для количества отсчетов Т = 2500 проводится в течение 1 с, т.е Т/Fs)
L = T; 
% Y = fft(M_pom_shum);
Y = fft(M_sign_no_isk+M_noise);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L; % шаг 

figure
plot(f/Fs, db(P1),'r','LineWidth',2)

xlim([0 (3/T_imp)]);
title('Signal spectrum')
xlabel('f (MHz)')
ylabel('|P1(f)|')

%% Построение спектра сигнала, поступившего на вход фильтра

Fs = 2500; % Гц
L = T;
% Y = fft(M_pom_shum);
Y = fft(M_Signal);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L; % шаг 

figure
plot(f/Fs, db(P1),'r','LineWidth',2)

xlim([0 (3/T_imp)]);
title('Построение спектра сигнала, поступившего на вход фильтра')
xlabel('f (MHz)')
ylabel('|P1(f)|')




