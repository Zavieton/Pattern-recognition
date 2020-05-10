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