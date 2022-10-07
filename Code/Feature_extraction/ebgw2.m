function ebgw2= ebgw2(seq1,L)
%len 可以调用主程序里面的lensec
seq1 = strrep(seq1,'X','');  % omit 'X'
seq1 = strrep(seq1,'Z','');  % omit 'Z'
seq1 = strrep(seq1,'B','');  % omit 'B'
seq1 = strrep(seq1,'O','');  % omit 'O'
seq1 = strrep(seq1,'J','');  % omit 'J'
seq1 = strrep(seq1,'U','');  % omit 'U'
%分组1
c= strrep(seq1,'A','X');
c1= strrep(c,'C','Z');
c2= strrep(c1,'D','B');
c3= strrep(c2,'E','B');
c4= strrep(c3,'I','X');
c5= strrep(c4,'K','O');
c6= strrep(c5,'L','X');
c7=strrep(c6,'M','X');
c8=strrep(c7,'N','Z');
 c9=strrep(c8,'P','X');
 c10=strrep(c9,'Q','Z');
 c11=strrep(c10,'R','O');
 c12=strrep(c11,'S','Z');
 c13=strrep(c12,'T','Z');
 c14=strrep(c13,'V','X');
 c15=strrep(c14,'W','X');
c16= strrep(c15,'Y','Z');
c17= strrep(c16,'G','X');
c18= strrep(c17,'H','O');
c19= strrep(c18,'F','X');
len=length(c19);
 %分组2
%  for i=1:len
%     if (seq1(i)=='G' || seq1(i)=='A' ||seq1(i)=='V' ||seq1(i)=='L' || seq1(i)=='I' ||seq1(i)=='M' ||seq1(i)=='P' || seq1(i)=='F' ||seq1(i)=='W')
%             seq2(i)='X';
%         elseif(seq1(i)=='Q' || seq1(i)=='N' ||seq1(i)=='S' ||seq1(i)=='T' || seq1(i)=='Y' ||seq1(i)=='C' )
%            seq2(i)='Z';
%         elseif(seq1(i)=='D' || seq1(i)=='E' )
%            seq2(i)='B';
%         elseif(seq1(i)=='H' || seq1(i)=='K'|| seq1(i)=='R' )
%            seq2(i)='O'; 
%     end
% end
 
%生成0-1映射
 seq2=zeros(1,len);
for i=1:len
    if (c19(i)=='X' || c19(i)=='B')
            seq2(i)=1;
        elseif(c19(i)=='Z' || c19(i)=='O')
           seq2(i)=0; 
    end
end
%检查0-1编码是否正确
% cc1=findstr(c19,'X');
% cc2=findstr(c19,'Z');
% cc3=findstr(c19,'B');
% cc4=findstr(c19,'O');
%求重量，把二进制序列变成长度是递增的L条子序列，k作为循环变量,控制行数，L是函数的输入变量
%新的序列的长度由列控制，（kn/L）取整是长度，即k*(len/L),取整fix
for k=1:L
    chang=fix(k*(len/L));
    seq3=seq2(1,1:chang);
    %计算该子序列1出现的频率
    ebgw2(1,k)=sum(seq3==1)/chang;
    clear chang seq3
end

