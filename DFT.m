function [An,Fn] = DFT(x)
fs=2000000; % ����Ƶ�ʣ��Լ�����ʵ���������
N=length(x); % x �Ǵ�����������
n=1:N;
%1-FFT
X=fft(x); % FFT
X=X(1:N/2);
Xabs=abs(X);
Xabs(1) = 0; %ֱ��������0
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
Fn = (k01+r1-1)*fs/N; %����Ƶ��
An= 2*pi*r1*Xabs(k01)/(N*sin(r1*pi)); %������ֵ