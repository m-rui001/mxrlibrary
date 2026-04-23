% number of realizations to generate
N = 2100;

% parameters for the Gaussian random field
gamma = 4;
tau = 5;
sigma = 25^(2);

% viscosity
visc = 0.01;

% grid size
s = 4096;
steps = 100;
nn = 101;
init_nn = 201;
input = zeros(N, init_nn);
output = zeros(N, steps+1, nn);
tspan = linspace(0,1,steps+1);
x = linspace(0, 1, init_nn);
X = linspace(0,1, nn);
if isempty(gcp('nocreate')) 
    parpool('local', 12); 
end

dq = parallel.pool.DataQueue;
afterEach(dq, @(data) incrementCounter());
parfor j=1:N
    u0 = GRF(s/2, 0, gamma, tau, sigma, "periodic");
    u = Burgers(u0, tspan, s, visc);
    
    u0_eval = u0(x);
    input(j,:) = u0_eval;

    temp_output = zeros(steps+1, nn);
    for k = 1:(steps+1)
        temp_output(k,:) = u{k}(X);
    end
    output(j,:,:) = temp_output;
    send(dq, 1);
end
save('Burger.mat', 'input', 'output', 'tspan', 'X', 'x')

function incrementCounter()
    persistent count;
    if isempty(count), count = 0; end
    count = count + 1;
    fprintf('\r当前进度: %d', count); 
end
