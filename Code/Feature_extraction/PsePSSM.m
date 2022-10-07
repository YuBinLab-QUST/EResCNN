function psepssm=PsePSSM(result_Na,lamdashu)
%读取蛋白质序列的pssm
WEISHU=numel(result_Na);
%所有蛋白质序列归一化
c=cell(WEISHU,1);
for t=1:WEISHU  
shu=result_Na{t};
%知道每条蛋白质的数量，提取出来的矩阵，注意蛋白质的顺序
% shuju=shu(1:i,1:20);
[M,N]=size(shu);
% shuju=shu(1:M-5,1:20);
shuju=shu;
d=[];
for i=1:M
   for j=1:20
       d(i,j)=1/(1+exp(-shuju(i,j)));
   end
end
% dd=shuju';
% d=zscore(dd);
% d=d';
c{t}=d(:,:);
end
%生成PSSM-AAC,x是一个,
for i=1:WEISHU
[MM,NN]=size(c{i});
 for  j=1:20
   x(i,j)=sum(c{i}(:,j))/MM;
 end
end
%PsePSSM后20*lamda
xx=[];
sheta=[];
shetaxin=[];
% lamda=1;
for lamda=1:lamdashu;
for t=1:WEISHU
  [MM,NN]=size(c{t});
  clear xx
   for  j=1:20
      for i=1:MM-lamda
       xx(i,j)=(c{t}(i,j)-c{t}(i+lamda,j))^2;
      end
      sheta(t,j)=sum(xx(1:MM-lamda,j))/(MM-lamda);
   end
end
shetaxin=[shetaxin,sheta];
end
psepssm=[x,shetaxin];