clear all;

rng(0);
muA = [-8;5];
muB = [-3;8];
theta = pi/4;
T = [cos(theta),-sin(theta); sin(theta), cos(theta)];
S = T*diag([1,2])*T'; %2x2
DataA = S*randn(2, 40) + muA * ones(1,40);
DataB = S*randn(2, 60) + muB * ones(1,60);

% training and testing data: each column is a sample
TrainA = DataA(:,1:end-20);
TestA = DataA(:,end-20+1:end);
TrainB = DataB(:,1:end-20);
TestB = DataB(:,end-20+1:end);

% show data points
figure(1)
hold on;
plot(TrainA(1,:),TrainA(2,:), '+r', 'DisplayName','ClassA');
plot(TrainB(1,:),TrainB(2,:), 'xb', 'DisplayName','ClassB');
axis equal; legend show; title('training data')

% sample mean & covariance
mA = mean(TrainA')';  %2x1
mB = mean(TrainB')';
sA = cov(TrainA');  %2x2
sB = cov(TrainB');

% separation vector
v = (sA+sB)\(mA-mB); %2x1
v = v/norm(v);

% show data
projA = v*v'*TrainA; %2x20
projB = v*v'*TrainB;
projmA = v*v'*mA; %2x1 
projmB = v*v'*mB;

figure(2)
hold on;
plot([TrainA(1,:),projA(1,:)], [TrainA(2,:),projA(2,:)], '+r');
plot([TrainB(1,:),projB(1,:)], [TrainB(2,:),projB(2,:)], 'xb');
plot([mA(1),projmA(1)],[mA(2),projmA(2)],'pr','MarkerFaceColor','red','Markersize',10);
plot([mB(1),projmB(1)],[mB(2),projmB(2)],'pb','MarkerFaceColor','blue','Markersize',10);
plot(7*[-v(1),v(1)], 7*[-v(2),v(2)], '-k');
axis equal; title('separation direction and projection')

c = v'*(mA+mB)/2;

classifyA = 0;
classifyB = 0;
missA = 0;
missB = 0;

for i=1:size(TestA,2)
    if v'*TestA(:,i) > c
        classifyA = classifyA+1;
            
    else
        missA = missA + 1;
    end
end

for i=1:size(TestB,2)
    if v'*TestB(:,i) <= c
        classifyB = classifyB+1;
        
    else
        missB = missB + 1;
    end
end

missRate = (missA+missB)/(size(TestA,2)+size(TestB,2));
successRate = 1 - missRate

load sonar.mat
whos('-file','sonar.mat');

load ionosphere.mat
whos('-file','ionosphere.mat');


sizeOfSonarA = sum(sonar_label == 0);
sizeOfSonarB = sum(sonar_label == 1);

sonarA = zeros(sizeOfSonarA, 60);
sonarB = zeros(sizeOfSonarB, 60);

%Store all sonar_data == 0 in Sonar A

for i = 1:size(sonar_label)
    if sonar_label(i) == 0
        sonarA(i,:) = sonar_data(i,:);
    end
end

ind = find(sum(sonarA,2)==0) ;
sonarA(ind,:) = [] ;    


for i = 1:size(sonar_label)
    if sonar_label(i) == 1
        sonarB(i,:) = sonar_data(i,:);
    end
end

%Deleting the 0 rows
ind = find(sum(sonarB,2)==0) ;
sonarB(ind,:) = [] ;    

%%%Create test and train data of sonar%%%

m = ceil(size(sonarA,1)*(7/10));

trainSonarA = sonarA(1:m,:);
testSonarA = sonarA(m+1:end,:);

m = ceil(size(sonarB,1)*(7/10));

trainSonarB = sonarB(1:m, :);
testSonarB = sonarB(m+1:end,:);

mSonarA = mean(trainSonarA)';
mSonarB = mean(trainSonarB)';
sSonarA = cov(trainSonarA);
sSonarB = cov(trainSonarB);

vSonar = (sSonarA+sSonarB)\(mSonarA-mSonarB);
vSonar = vSonar/norm(vSonar);

cSonar = -0.033;

classifySonarA = 0;
classifySonarB = 0;
missSonarA = 0;
missSonarB = 0;

for i=1:size(testSonarA,1)
    if vSonar'*testSonarA(i,:)' > cSonar
        classifySonarA = classifySonarA+1;
            
    else
        missSonarA = missSonarA + 1;
    end
end

for i=1:size(testSonarB,1)
    if vSonar'*testSonarB(i,:)' <= cSonar
        classifySonarB = classifySonarB+1;
            
    else
        missSonarB = missSonarB + 1;
    end
end

missRateSonar = (missSonarA+missSonarB)/(size(testSonarA,1)+size(testSonarB,1));
successRateSonar = 1 - missRateSonar

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

cIono = 0.5;

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
g1SA = vIono'*mIonoA
g2SA = vIono'*mIonoB