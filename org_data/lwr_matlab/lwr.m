% In this example, all units are SI: meters, seconds.
% The algorithm itself is unit-agnostic, and only needs the unit used as
% input to be consistent.

%This creates a Greenshields fundamental diagram
% Free flow speed 30 m/s (67.1mph)
% Maximum Density 0.1 veh / m (100 vehicles/km, 10 meters per vehicle)
fd = LH_Greenshields(30, .1);

%This creates a spatial domain between 0 and 1000, where the fundamental
%diagram is fd between 0 and 1000.
pbEnv = LH_general(fd, 0, 1000);

%Input arrays for initial densities.
%a - locations: x =0, 2, 4, ..., 100
%b - initial density: k = 0.1, 0.0998, 0.0996, ..., 0
%the third comment removes last density k = 0, so "a" has 501 values, and 
%"b" has 500 values
a = linspace(0, 1000, 501);
b = 0.1 - 0.1 / 1000 * a;
b(end) = [];

%set initial density, "a" gives locations, "b" specifies the density
%between the locations
% pbEnv.setIniDens([0 20 50 1000], [80E-3 10E-3 30E-3]);
pbEnv.setIniDens([a], [b]);

%Sample arrays for upstream flows.
%Upstream Flows: 0 veh/s at 0<t<50s
pbEnv.setUsFlows([0 50], [0]);

%Sample arrays for downstream flows.
%Downstream flows: 0 veh/s at 0<t<50s
pbEnv.setDsFlows([0 50], [0]);


%Here we create two matrices tValues and xValues which store space and time
%information for every point at which we want to compute the solution.
X = 1000;           % Maximal x in the computational domain
nx = 500;           % Number x grid points
T = 50;            % Maximal time for the computation
nt = 500;           % Number of t grid points
dx=X/nx;            % Space step
dt=T/nt;            % Time step
xScale = 0:dx:X;    % Create vector array for spatial domain
tScale = 0:dt:T;    % Create vector array for temporal domain

% Explication matrices pour donner l information (3 lignes max)
xValues = ones(size(tScale'))*(xScale);
tValues = tScale' * ones(size(xScale));

% tic/toc or [cputime] can be used to time the computation
tic
result = pbEnv.explSol(tValues,xValues);

% result{1} is the Moskowitz function 
N = result{1};

% result{2} is the active component matrix (needed for computation of density)
activeComp = result{2};

k = pbEnv.density(tValues,xValues,activeComp);%matrix of solution densities
toc

LH_plot3D(tScale, xScale, N, k, fd)
figure
LH_plot2D(tScale, xScale, N, k, fd)
save('lwr')
csvwrite('density.csv', k)