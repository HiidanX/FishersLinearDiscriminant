clear all;
load ionosphere.mat

%%% Same for Ionosphere %%%

sizeOfIonoA = sum(ionosphere_label == 0);
sizeOfIonoB = sum(ionosphere_label == 1);

ionoA = zeros(sizeOfIonoA, size(ionosphere_data,2));
ionoB = zeros(sizeOfIonoB, size(ionosphere_data,2));

%Store all ionosphere_data == 0 in Sonar A

for i = 1:size(ionosphere_label)
    if ionosphere_label(i) == 0
        ionoA(i,:) = ionosphere_data(i,:);
    end
end

ind = find(sum(ionoA,2)==0) ;
ionoA(ind,:) = [] ;    


for i = 1:size(ionosphere_label)
    if ionosphere_label(i) == 1
        ionoB(i,:) = ionosphere_data(i,:);
    end
end

ind = find(sum(ionoB,2)==0);
ionoB(ind,:) = []; 



m = ceil(size(ionoA,1)*(7/10));

trainIonoA = ionoA(1:m,:);
testIonoA = ionoA(m+1:end,:);

m = ceil(size(ionoB,1)*(7/10));

trainIonoB = ionoB(1:m, :);
testIonoB = ionoB(m+1:end,:);

mIonoA = mean(trainIonoA)';
mIonoB = mean(trainIonoB)';
sIonoA = cov(trainIonoA);
sIonoB = cov(trainIonoB);

vIono = pinv(sIonoA+sIonoB)*(mIonoA-mIonoB);
vIono = vIono/norm(vIono);

cIono = vIono'*(mIonoA + mIonoB)/2;

classifyIonoA = 0;
classifyIonoB = 0;
missIonoA = 0;
missIonoB = 0;

for i=1:size(testIonoA,1)
    if vIono'*testIonoA(i,:)' > cIono
        classifyIonoA = classifyIonoA+1;
            
    else
        missIonoA = missIonoA + 1;
    end
end

for i=1:size(testIonoB,1)
    if vIono'*testIonoB(i,:)' <= cIono
        classifyIonoB = classifyIonoB+1;
            
    else
        missIonoB = missIonoB + 1;
    end
end

missRateIono = (missIonoA+missIonoB)/(size(testIonoA,1)+size(testIonoB,1));
successRateIono = 1 - missRateIono